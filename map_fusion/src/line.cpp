#include "line.h"
#include <iostream>

using namespace std;
using namespace Eigen;
line2d::line2d(const Eigen::Vector4d &vec)
{
    ptstart = Vector2d(vec[0], vec[1]);
    ptend = Vector2d(vec[2], vec[3]);
    ptmid = (ptstart + ptend) / 2;
    subvec = ptend - ptstart;
    length = subvec.norm();
    direction = subvec / length;
    hptstart = Vector3d(ptstart[0],ptstart[1], 1);
    hptend = Vector3d(ptend[0],ptend[1], 1);
    A = ptend[1] - ptstart[1];
    B = ptstart[0] - ptend[0];
    C = ptend[0] * ptstart[1] - ptstart[0] * ptend[1];
    A2B2 = sqrt(A * A + B * B);
}

Eigen::Vector3d line2d::ComputeCircleNormal(const Eigen::Matrix3d &K)
{
    Vector3d p1_ = K.inverse() * hptstart;
    Vector3d p2_ = K.inverse() * hptend;
    MatrixXd A(3, 4);
    A << 0, 0, 0, 1,
        p1_.transpose(), 1,
        p2_.transpose(), 1;
    EigenSolver<MatrixXd> es(A.transpose() * A);
    Vector4d v = es.eigenvectors().col(2).real();
    if (es.eigenvalues()[2].real() > es.eigenvalues()[3].real())
        v = es.eigenvectors().col(3).real();
    // cout << "matrix:" << A << endl
    //      << "eigenvalues:" << es.eigenvalues() << endl
    //      << "vec:" << endl
    //      << v << endl;
    Vector3d normal(v[0], v[1], v[2]);
    return normal.normalized();
}

Eigen::Vector2d line2d::point2flined(const Eigen::Vector2d &mdpt)
{
    Vector2d tmpstart = mdpt - ptstart;
    double d1 = tmpstart.norm();
    Vector2d tmpend = mdpt - ptend;
    double d2 = tmpend.norm();
 
    // line Bx-Ay+C2=0
    double A2, B2, C2;
    A2 = B;
    B2 = -A;
    C2 = -(A2 * mdpt[0] + B2 * mdpt[1]);
    Matrix2d Cof;
    Cof << A, B, A2, B2;
    Vector2d intersection = Cof.inverse() * Vector2d(-C, -C2);
    if((intersection.x()-ptstart.x())*(intersection.x()-ptend.x())>=0)
    {
        if (d1 < d2) return ptstart;
        else
          return ptend;
    }
    else
    {
        return intersection;
    }
}

line3d::line3d(const Vector6d &vec)
{
    ptstart = Vector3d(vec[0], vec[1], vec[2]);
    ptend = Vector3d(vec[3], vec[4], vec[5]);
    ptmid = (ptstart + ptend) / 2;
    Vector3d temp = ptend - ptstart;
    length = temp.norm();
    direction = temp / length;
    //hptstart = Vector4d(ptstart, 1);
    //hptend = Vector4d(ptend, 1);
}

line3d::line3d(const Eigen::Vector3d &_ptstart, const Eigen::Vector3d &_ptend)
{
    ptstart = _ptstart;
    ptend = _ptend;
    ptmid = (ptstart + ptend) / 2;
    Vector3d temp = ptend - ptstart;
    length = temp.norm();
    direction = temp / length;
}
line3d line3d::transform3D(const Eigen::Matrix3d &R, const Eigen::Vector3d &t)
{
    line3d transformedLine3d;
    //cout<<"before transform:"<<ptstart[0]<<","<<ptstart[1]<<","<<ptstart[2]<<endl;
    Vector3d tempstart = R * ptstart + t;
    Vector3d tempend = R * ptend + t;
    //cout<<"After tmptransform:"<<tempstart[0]<<","<<tempstart[1]<<","<<tempstart[2]<<endl;
    Vector6d tempvec;
    tempvec.block(0,0,3,1)=tempstart;
    tempvec.block(3,0,3,1)=tempend;
    line3d transform3dline(tempvec);
    //cout<<"After transform:"<<transform3dline.ptstart[0]<<","<<transform3dline.ptstart[1]<<","<<transform3dline.ptstart[2]<<endl;
    return transform3dline;
}

line2d line3d::project3D(const Eigen::Matrix3d &K)
{
    Vector3d tmpstart=K*ptstart;
    double v1= tmpstart[0]/tmpstart[2];
    double v2= tmpstart[1]/tmpstart[2];
    Vector3d tmpend=K*ptend;
    double v3= tmpend[0]/tmpend[2];
    double v4= tmpend[1]/tmpend[2];
    line2d projectedline(Vector4d(v1,v2,v3,v4));
    return projectedline;
}

pairsmatch::pairsmatch(int indx, line2d tmp2d, line3d tmp3d)
{
    index = indx;
    line2dt = tmp2d;
    line3dt = tmp3d;
    distance<< 2000,1000,1000;
}
// Follow Brown, Mark, David Windridge, and Jean-Yves Guillemaut. 
//"A family of globally optimal branch-and-bound algorithms for 2D–3D correspondence-free registration."
// Pattern Recognition 93 (2019): 36-54.
Eigen::Vector3d pairsmatch::calcAngleDist(const Eigen::Matrix3d &K, const double & lamda)
{
    // the angle distance between two orientations
    
    Eigen::Vector3d circleNormal;
    circleNormal=line2dt.ComputeCircleNormal(K);
    double theta=abs(PI/2-acos(abs(line3dt.direction.transpose()*circleNormal)));

    if(theta>0.12) return Vector3d{PI, PI, 0}; //if the inter angle is too large, they should not be correspondences
    //mid 2D point to the latest point angle of 3D line projection (finite 3D line)
    Vector3d hmid=Vector3d(line2dt.ptmid[0],line2dt.ptmid[1],1);
    Vector3d bearing_vector=(K.inverse()*hmid).normalized();

    line2d project2dline=line3dt.project3D(K);
    // add penalty for large length error and non-overlap
    if (project2dline.length < 0.5 * line2dt.length)
        return Vector3d{PI, PI, 0};
    if(line2dt.point2flined(project2dline.ptstart)==line2dt.point2flined(project2dline.ptend))  //no overlap
    {
        return Vector3d{PI, PI, 0};
    }    
    
    Vector2d interpt=project2dline.point2flined(line2dt.ptmid);
    Vector3d hinterpt=Vector3d(interpt[0],interpt[1],1);
    Vector3d bearing_vector_interpt=(K.inverse()*hinterpt).normalized();
    double phi=acos(abs(bearing_vector.transpose()*bearing_vector_interpt));
    //printmsg();
    //cout<<"theta="<<theta<<", phi="<<phi<<endl;
    double dist = lamda*theta+(1-lamda)*phi;
    return Vector3d{dist, theta, phi};
}

Eigen::Vector3d pairsmatch::calcEulerDist(const Eigen::Matrix3d &K, const double & theta)
{
    double d1=500, d2=500, dist=1000;
    line2d project2dline=line3dt.project3D(K);

    // if(line3dt.ptmid[2]>15.0)
    //     return Vector3d{dist, 3.14, 0};
    
    double angle=acos(abs(line2dt.direction.transpose()*project2dline.direction));
    double overlap_dist=0;

    if (angle < theta) //10 degree 0.1745
    {
        // add penalty for large length error and non-overlap
        if (project2dline.length < 0.5 * line2dt.length)
            return Vector3d{dist, 3.14, 0};
        //overlap distance
        overlap_dist=(line2dt.point2flined(project2dline.ptstart)-line2dt.point2flined(project2dline.ptend)).norm();
        if(overlap_dist<0.5*min(project2dline.length, line2dt.length))
            return Vector3d{dist, 3.14, 0};
        d1 = abs(line2dt.A * project2dline.ptstart[0] + line2dt.B * project2dline.ptstart[1] + line2dt.C) / line2dt.A2B2;
        
        d2 = abs(line2dt.A * project2dline.ptend[0] + line2dt.B * project2dline.ptend[1] + line2dt.C) / line2dt.A2B2;
        
        dist = d1 + d2;
    }
    // cout<<theta<<" dist="<<dist<<"dist1="<<d1<<", dist2="<<d2<<endl;
    return Vector3d{dist, angle, overlap_dist};
}
void pairsmatch::printmsg()
{
    cout<<index<<"-th pairs, 3d("<<line3dt.ptstart[0]<<","<<line3dt.ptstart[1]<<","<<line3dt.ptstart[2]<<","<<line3dt.ptstart[3]<<","<<line3dt.ptstart[4]<<","<<line3dt.ptstart[5]<<")"<<endl;
    cout<<"2d("<<line2dt.ptstart[0]<<","<<line2dt.ptstart[1]<<","<<line2dt.ptend[0]<<","<<line2dt.ptend[1]<<")"<<endl;
}

//reject outliers
bool rejectoutliers(std::vector<pairsmatch> &matches,
                    const Eigen::Matrix3d &K,
                    const Eigen::Matrix3d &R,
                    const Eigen::Vector3d &t, double &lamda, double &outlier_threshold,
                    bool &UseAngleDist)
{
    vector<pairsmatch> updatematches;
    for (size_t i = 0; i < matches.size(); i++)
    {
        Vector3d dist;
        line3d tmpline3d = matches[i].line3dt;
        pairsmatch tfmatch(i, matches[i].line2dt, tmpline3d.transform3D(R, t));
        if (UseAngleDist)
            dist = tfmatch.calcAngleDist(K, lamda);
        else
        {
            dist = tfmatch.calcEulerDist(K, lamda);
        }

        //cout << i << "-th dist=" << dist << endl;
        if (dist.x() <= outlier_threshold)
        {
            matches[i].distance=dist;
            updatematches.push_back(matches[i]);
        }
    }
    bool hasoutliers = false;
    if (updatematches.size() < matches.size())
        hasoutliers = true;
    else
    {
        hasoutliers = false;
    }
    matches=updatematches;
    return hasoutliers;
}

bool GreaterSort(pairsmatch pair1,pairsmatch pair2 )
{
    return (pair1.distance.z()>pair2.distance.z());     
}
//update correspondence from the oringinal 2D and 3D lines
std::vector<pairsmatch> updatecorrespondence(std::vector<line3d> &lines_3d,
                                             std::vector<line2d> &lines_2d,
                                             const Eigen::Matrix3d &K,
                                             const Eigen::Matrix3d &R,
                                             const Eigen::Vector3d &t,
                                             double &theta, double &outlier_threshold)
{
    vector<pairsmatch> updatemaches;
    int indx=0;
    for (size_t i = 0; i < lines_2d.size(); i++)
    {
        int index = 0;
        double mindist = 1000;
        Vector3d vecdist{0,0,0};
        for (size_t j = 0; j < lines_3d.size(); j++)
        {
            pairsmatch tfmatch(i, lines_2d[i], lines_3d[j].transform3D(R, t));
            Vector3d dist = tfmatch.calcEulerDist(K, theta);
            if (dist.x() < mindist)
            {
                mindist = dist.x();
                vecdist=dist;
                index = j;
            }
        }
        if (mindist<outlier_threshold)
        {
            // cout<<i<<"-th line2d is matched with "<<index<<"-th line3d, mindist="<<mindist<<endl;
            pairsmatch match(indx, lines_2d[i], lines_3d[index]);
            match.distance=vecdist;
            updatemaches.push_back(match);
            indx++;
        }
    }
    //select the former correspondences with largest overlap distance
    if (1)
    {
        sort(updatemaches.begin(), updatemaches.end(), GreaterSort);
        if (updatemaches.size() > 40) //only keep former 30 correspondences
            updatemaches.erase(updatemaches.begin() + 40, updatemaches.end());
    }
    return updatemaches;
}

double computedistribution(const std::vector<pairsmatch> &matches)
{
    double confidence=0.0;
    Eigen::MatrixXd Vec(matches.size(),3);
    for (size_t i=0; i<matches.size(); i++)
    {
        Eigen::Vector3d dir=matches[i].line3dt.direction;
        Vec.row(i) <<dir.x(), dir.y(), dir.z();
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Vec, Eigen::ComputeFullV | Eigen::ComputeFullU); 
	Eigen::MatrixXd singular_values = svd.singularValues();
	// Eigen::MatrixXf left_singular_vectors = svd.matrixU();
	// Eigen::MatrixXf right_singular_vectors = svd.matrixV();
    
    confidence=singular_values(singular_values.rows()-1);
    return confidence;
}