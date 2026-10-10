// NOLINTBEGIN
/*
    MIT License

    Copyright (c) 2021 Zhepei Wang (wangzhepei@live.com)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

#ifndef TRAJECTORY_HPP
#define TRAJECTORY_HPP

#include <Eigen/Eigen>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <vector>

#include "xgc2_math/optimization/root_finder.hpp"

namespace xgc2_math::optimization {

/**
 * @brief 轨迹片段类模板
 * @tparam D 多项式的阶数（degree）
 *
 * 该类表示一段多项式轨迹，支持3维空间中的位置、速度、加速度和Jerk计算
 * 多项式形式为: p(t) = c0 + c1*t + c2*t^2 + ... + cD*t^D
 */

template <int D> class Piece {
  public:
    // 类型定义：位置系数矩阵，3行 × (D+1)列，存储多项式系数 [c0, c1, ..., cD]
    typedef Eigen::Matrix<double, 3, D + 1> CoefficientMat;
    // 类型定义：速度系数矩阵，3行 × D列
    typedef Eigen::Matrix<double, 3, D> VelCoefficientMat;
    // 类型定义：加速度系数矩阵，3行 × (D-1)列
    typedef Eigen::Matrix<double, 3, D - 1> AccCoefficientMat;

  private:
    double duration{0.0};                            // 该轨迹片段的持续时间
    CoefficientMat coeffMat{CoefficientMat::Zero()}; // 系数矩阵，每行对应x, y, z三个维度的多项式系数

  public:
    // 默认构造函数
    Piece() = default;

    /**
     * @brief 构造函数
     * @param dur 轨迹片段的持续时间
     * @param cMat 系数矩阵
     */
    Piece(double dur, const CoefficientMat& cMat) : duration(dur), coeffMat(cMat) {}

    /**
     * @brief 获取轨迹的空间维度
     * @return 维度数（固定为3，表示3D空间）
     */
    inline int getDim() const { return 3; }

    /**
     * @brief 获取多项式的阶数
     * @return 多项式阶数D
     */
    inline int getDegree() const { return D; }

    /**
     * @brief 获取轨迹片段的持续时间
     * @return 持续时间（秒）
     */
    inline double getDuration() const { return duration; }

    /**
     * @brief 获取系数矩阵的常量引用
     * @return 系数矩阵
     */
    inline const CoefficientMat& getCoeffMat() const { return coeffMat; }

    /**
     * @brief 计算给定时刻的位置
     * @param t 时间参数（0 <= t <= duration）
     * @return 3D位置向量
     *
     * 使用Horner方法高效计算多项式值: p(t) = c0 + c1*t + c2*t^2 + ... + cD*t^D
     */
    inline Eigen::Vector3d getPos(const double& t) const {
        Eigen::Vector3d pos(0.0, 0.0, 0.0);
        double tn = 1.0; // t的n次方，从t^0开始
        for (int i = D; i >= 0; i--) {
            pos += tn * coeffMat.col(i); // 累加每一项
            tn *= t;                     // 更新t的幂次
        }
        return pos;
    }

    /**
     * @brief 计算给定时刻的速度（位置对时间的一阶导数）
     * @param t 时间参数（0 <= t <= duration）
     * @return 3D速度向量
     *
     * 速度是位置的导数: v(t) = dp/dt = c1 + 2*c2*t + 3*c3*t^2 + ... + D*cD*t^(D-1)
     */
    inline Eigen::Vector3d getVel(const double& t) const {
        Eigen::Vector3d vel(0.0, 0.0, 0.0);
        double tn = 1.0; // t的n次方
        int n = 1;       // 当前项的系数倍数
        for (int i = D - 1; i >= 0; i--) {
            vel += n * tn * coeffMat.col(i); // n是导数带来的系数
            tn *= t;
            n++;
        }
        return vel;
    }

    /**
     * @brief 计算给定时刻的加速度（位置对时间的二阶导数）
     * @param t 时间参数（0 <= t <= duration）
     * @return 3D加速度向量
     *
     * 加速度是速度的导数: a(t) = dv/dt = 2*c2 + 6*c3*t + ... + D*(D-1)*cD*t^(D-2)
     */
    inline Eigen::Vector3d getAcc(const double& t) const {
        Eigen::Vector3d acc(0.0, 0.0, 0.0);
        double tn = 1.0; // t的n次方
        int m = 1;       // 第一个系数倍数
        int n = 2;       // 第二个系数倍数
        for (int i = D - 2; i >= 0; i--) {
            acc += m * n * tn * coeffMat.col(i); // m*n是二阶导数带来的系数
            tn *= t;
            m++;
            n++;
        }
        return acc;
    }

    /**
     * @brief 计算给定时刻的Jerk（位置对时间的三阶导数）
     * @param t 时间参数（0 <= t <= duration）
     * @return 3D Jerk向量
     *
     * Jerk是加速度的导数: j(t) = da/dt = 6*c3 + 24*c4*t + ... + D*(D-1)*(D-2)*cD*t^(D-3)
     */
    inline Eigen::Vector3d getJer(const double& t) const {
        Eigen::Vector3d jer(0.0, 0.0, 0.0);
        double tn = 1.0; // t的n次方
        int l = 1;       // 第一个系数倍数
        int m = 2;       // 第二个系数倍数
        int n = 3;       // 第三个系数倍数
        for (int i = D - 3; i >= 0; i--) {
            jer += l * m * n * tn * coeffMat.col(i); // l*m*n是三阶导数带来的系数
            tn *= t;
            l++;
            m++;
            n++;
        }
        return jer;
    }

    /**
     * @brief 归一化位置系数矩阵
     * @return 归一化后的系数矩阵
     *
     * 将多项式从参数t转换为归一化参数s（s = t/duration），使得s∈[0,1]
     * 转换公式: p(t) = p(s*T) = Σ c_i * (s*T)^i = Σ (c_i * T^i) * s^i
     */
    inline CoefficientMat normalizePosCoeffMat() const {
        CoefficientMat nPosCoeffsMat;
        double t = 1.0; // duration的幂次
        for (int i = D; i >= 0; i--) {
            nPosCoeffsMat.col(i) = coeffMat.col(i) * t; // 乘以duration^i
            t *= duration;
        }
        return nPosCoeffsMat;
    }

    /**
     * @brief 归一化速度系数矩阵
     * @return 归一化后的速度系数矩阵
     *
     * 将速度多项式转换为归一化参数s的形式
     * v(t) = dp/dt = dp/ds * ds/dt = (1/T) * dp/ds
     */
    inline VelCoefficientMat normalizeVelCoeffMat() const {
        VelCoefficientMat nVelCoeffMat;
        int n = 1;           // 导数系数
        double t = duration; // duration的幂次
        for (int i = D - 1; i >= 0; i--) {
            nVelCoeffMat.col(i) = n * coeffMat.col(i) * t; // n * c_i * T^i
            t *= duration;
            n++;
        }
        return nVelCoeffMat;
    }

    /**
     * @brief 归一化加速度系数矩阵
     * @return 归一化后的加速度系数矩阵
     *
     * 将加速度多项式转换为归一化参数s的形式
     * a(t) = d²p/dt² = d²p/ds² * (ds/dt)² = (1/T²) * d²p/ds²
     */
    inline AccCoefficientMat normalizeAccCoeffMat() const {
        AccCoefficientMat nAccCoeffMat;
        int n = 2;                      // 第二个导数系数
        int m = 1;                      // 第一个导数系数
        double t = duration * duration; // duration的平方
        for (int i = D - 2; i >= 0; i--) {
            nAccCoeffMat.col(i) = n * m * coeffMat.col(i) * t; // n*m * c_i * T^(i+2)
            n++;
            m++;
            t *= duration;
        }
        return nAccCoeffMat;
    }

    /**
     * @brief 计算轨迹片段的最大速度
     * @return 最大速度的模长
     *
     * 通过求解速度模长平方的导数的根来找到极值点，然后比较所有候选点的速度
     * 算法步骤：
     * 1. 构造速度模长平方的多项式 ||v(t)||² = vx² + vy² + vz²
     * 2. 对该多项式求导，找到所有根（极值点）
     * 3. 比较所有极值点和端点的速度，取最大值
     */
    inline double getMaxVelRate() const {
        VelCoefficientMat nVelCoeffMat = normalizeVelCoeffMat();
        // 构造速度模长平方的多项式系数：||v||² = vx² + vy² + vz²
        Eigen::VectorXd coeff = RootFinder::polySqr(nVelCoeffMat.row(0)) + RootFinder::polySqr(nVelCoeffMat.row(1)) +
                                RootFinder::polySqr(nVelCoeffMat.row(2));
        int N = static_cast<int>(coeff.size());
        int n = N - 1;
        // 对||v||²求导，得到d(||v||²)/dt的系数
        for (int i = 0; i < N; i++) {
            coeff(i) *= n;
            n--;
        }
        // 如果导数接近零，说明速度恒定
        if (coeff.head(N - 1).squaredNorm() < DBL_EPSILON) {
            return getVel(0.0).norm();
        } else {
            // 设置求根区间的初始边界
            double l = -0.0625;
            double r = 1.0625;
            // 调整左边界，确保多项式在该点不为零
            while (fabs(RootFinder::polyVal(coeff.head(N - 1), l)) < DBL_EPSILON) {
                l = 0.5 * l;
            }
            // 调整右边界，确保多项式在该点不为零
            while (fabs(RootFinder::polyVal(coeff.head(N - 1), r)) < DBL_EPSILON) {
                r = 0.5 * (r + 1.0);
            }
            // 求解导数多项式的所有根（极值点）
            std::set<double> candidates = RootFinder::solvePolynomial(coeff.head(N - 1), l, r, FLT_EPSILON / duration);
            // 添加起点和终点作为候选
            candidates.insert(0.0);
            candidates.insert(1.0);
            double maxVelRateSqr = -INFINITY;
            double tempNormSqr;
            // 遍历所有候选点，找到最大速度
            for (std::set<double>::const_iterator it = candidates.begin(); it != candidates.end(); ++it) {
                if (0.0 <= *it && 1.0 >= *it) {
                    tempNormSqr = getVel((*it) * duration).squaredNorm();
                    maxVelRateSqr = maxVelRateSqr < tempNormSqr ? tempNormSqr : maxVelRateSqr;
                }
            }
            return sqrt(maxVelRateSqr);
        }
    }

    /**
     * @brief 计算轨迹片段的最大加速度
     * @return 最大加速度的模长
     *
     * 算法与getMaxVelRate()类似，通过求解加速度模长平方的导数的根来找到极值点
     * 算法步骤：
     * 1. 构造加速度模长平方的多项式 ||a(t)||² = ax² + ay² + az²
     * 2. 对该多项式求导，找到所有根（极值点）
     * 3. 比较所有极值点和端点的加速度，取最大值
     */
    inline double getMaxAccRate() const {
        AccCoefficientMat nAccCoeffMat = normalizeAccCoeffMat();
        // 构造加速度模长平方的多项式系数：||a||² = ax² + ay² + az²
        Eigen::VectorXd coeff = RootFinder::polySqr(nAccCoeffMat.row(0)) + RootFinder::polySqr(nAccCoeffMat.row(1)) +
                                RootFinder::polySqr(nAccCoeffMat.row(2));
        int N = static_cast<int>(coeff.size());
        int n = N - 1;
        // 对||a||²求导
        for (int i = 0; i < N; i++) {
            coeff(i) *= n;
            n--;
        }
        // 如果导数接近零，说明加速度恒定
        if (coeff.head(N - 1).squaredNorm() < DBL_EPSILON) {
            return getAcc(0.0).norm();
        } else {
            // 设置求根区间的初始边界
            double l = -0.0625;
            double r = 1.0625;
            // 调整左边界
            while (fabs(RootFinder::polyVal(coeff.head(N - 1), l)) < DBL_EPSILON) {
                l = 0.5 * l;
            }
            // 调整右边界
            while (fabs(RootFinder::polyVal(coeff.head(N - 1), r)) < DBL_EPSILON) {
                r = 0.5 * (r + 1.0);
            }
            // 求解导数多项式的所有根
            std::set<double> candidates = RootFinder::solvePolynomial(coeff.head(N - 1), l, r, FLT_EPSILON / duration);
            // 添加端点
            candidates.insert(0.0);
            candidates.insert(1.0);
            double maxAccRateSqr = -INFINITY;
            double tempNormSqr;
            // 遍历所有候选点，找到最大加速度
            for (std::set<double>::const_iterator it = candidates.begin(); it != candidates.end(); ++it) {
                if (0.0 <= *it && 1.0 >= *it) {
                    tempNormSqr = getAcc((*it) * duration).squaredNorm();
                    maxAccRateSqr = maxAccRateSqr < tempNormSqr ? tempNormSqr : maxAccRateSqr;
                }
            }
            return sqrt(maxAccRateSqr);
        }
    }

    /**
     * @brief 检查轨迹片段是否满足最大速度约束
     * @param maxVelRate 最大允许速度
     * @return 如果满足约束返回true，否则返回false
     *
     * 算法：
     * 1. 首先检查端点速度是否超限
     * 2. 然后检查||v(t)||² - maxVelRate² 在[0,1]区间内是否有根
     * 3. 如果有根，说明存在某个时刻速度超限
     */
    inline bool checkMaxVelRate(const double& maxVelRate) const {
        double sqrMaxVelRate = maxVelRate * maxVelRate;
        // 首先检查起点和终点的速度
        if (getVel(0.0).squaredNorm() >= sqrMaxVelRate || getVel(duration).squaredNorm() >= sqrMaxVelRate) {
            return false;
        } else {
            VelCoefficientMat nVelCoeffMat = normalizeVelCoeffMat();
            // 构造||v||²的多项式
            Eigen::VectorXd coeff = RootFinder::polySqr(nVelCoeffMat.row(0)) +
                                    RootFinder::polySqr(nVelCoeffMat.row(1)) + RootFinder::polySqr(nVelCoeffMat.row(2));
            double t2 = duration * duration;
            // 减去maxVelRate²，得到||v||² - maxVelRate²的多项式
            coeff.tail<1>()(0) -= sqrMaxVelRate * t2;
            // 检查该多项式在[0,1]区间内是否有根（即是否存在超速点）
            return RootFinder::countRoots(coeff, 0.0, 1.0) == 0;
        }
    }

    /**
     * @brief 检查轨迹片段是否满足最大加速度约束
     * @param maxAccRate 最大允许加速度
     * @return 如果满足约束返回true，否则返回false
     *
     * 算法与checkMaxVelRate()类似：
     * 1. 首先检查端点加速度是否超限
     * 2. 然后检查||a(t)||² - maxAccRate² 在[0,1]区间内是否有根
     * 3. 如果有根，说明存在某个时刻加速度超限
     */
    inline bool checkMaxAccRate(const double& maxAccRate) const {
        double sqrMaxAccRate = maxAccRate * maxAccRate;
        // 首先检查起点和终点的加速度
        if (getAcc(0.0).squaredNorm() >= sqrMaxAccRate || getAcc(duration).squaredNorm() >= sqrMaxAccRate) {
            return false;
        } else {
            AccCoefficientMat nAccCoeffMat = normalizeAccCoeffMat();
            // 构造||a||²的多项式
            Eigen::VectorXd coeff = RootFinder::polySqr(nAccCoeffMat.row(0)) +
                                    RootFinder::polySqr(nAccCoeffMat.row(1)) + RootFinder::polySqr(nAccCoeffMat.row(2));
            double t2 = duration * duration;
            double t4 = t2 * t2;
            // 减去maxAccRate²，得到||a||² - maxAccRate²的多项式
            coeff.tail<1>()(0) -= sqrMaxAccRate * t4;
            // 检查该多项式在[0,1]区间内是否有根
            return RootFinder::countRoots(coeff, 0.0, 1.0) == 0;
        }
    }
};

/**
 * @brief 轨迹类模板，由多个轨迹片段组成的完整轨迹
 * @tparam D 多项式的阶数（degree）
 *
 * 该类管理一条由多个Piece组成的完整轨迹，提供统一的接口来查询轨迹的位置、速度、加速度等信息
 */
template <int D> class Trajectory {
  private:
    typedef std::vector<Piece<D>> Pieces; // 轨迹片段的容器类型
    Pieces pieces;                        // 存储所有轨迹片段的向量

  public:
    // 默认构造函数
    Trajectory() = default;

    /**
     * @brief 从持续时间和系数矩阵构造轨迹
     * @param durs 每个片段的持续时间向量
     * @param cMats 每个片段的系数矩阵向量
     *
     * 构造函数会根据输入创建相应数量的轨迹片段
     */
    Trajectory(const std::vector<double>& durs, const std::vector<typename Piece<D>::CoefficientMat>& cMats) {
        int N = static_cast<int>(std::min(durs.size(), cMats.size())); // 取较小的尺寸
        pieces.reserve(static_cast<typename Pieces::size_type>(N));
        for (int i = 0; i < N; i++) {
            pieces.emplace_back(durs[i], cMats[i]);
        }
    }

    /**
     * @brief 获取轨迹片段的数量
     * @return 片段数量
     */
    inline int getPieceNum() const { return static_cast<int>(pieces.size()); }

    /**
     * @brief 获取所有片段的持续时间
     * @return 持续时间向量，大小为片段数量
     */
    inline Eigen::VectorXd getDurations() const {
        int N = getPieceNum();
        Eigen::VectorXd durations(N);
        for (int i = 0; i < N; i++) {
            durations(i) = pieces[static_cast<typename Pieces::size_type>(i)].getDuration();
        }
        return durations;
    }

    /**
     * @brief 获取整条轨迹的总持续时间
     * @return 总时间（所有片段时间之和）
     */
    inline double getTotalDuration() const {
        int N = getPieceNum();
        double totalDuration = 0.0;
        for (int i = 0; i < N; i++) {
            totalDuration += pieces[i].getDuration();
        }
        return totalDuration;
    }

    /**
     * @brief 获取所有连接点（waypoints）的位置
     * @return 3×(N+1)矩阵，包含N个片段的N+1个连接点位置
     *
     * 返回所有轨迹片段的起点和终点位置
     */
    inline Eigen::Matrix3Xd getPositions() const {
        int N = getPieceNum();
        Eigen::Matrix3Xd positions(3, N + 1);
        // 获取每个片段的起点（即系数矩阵的最后一列，即t=0时的位置）
        for (int i = 0; i < N; i++) {
            positions.col(i) = pieces[i].getCoeffMat().col(D);
        }
        // 最后一个点是最后一个片段的终点
        positions.col(N) = pieces[N - 1].getPos(pieces[N - 1].getDuration());
        return positions;
    }

    /**
     * @brief 常量下标运算符，访问指定索引的轨迹片段
     * @param i 片段索引
     * @return 第i个片段的常量引用
     */
    inline const Piece<D>& operator[](int i) const { return pieces[static_cast<typename Pieces::size_type>(i)]; }

    /**
     * @brief 下标运算符，访问指定索引的轨迹片段
     * @param i 片段索引
     * @return 第i个片段的引用
     */
    inline Piece<D>& operator[](int i) { return pieces[static_cast<typename Pieces::size_type>(i)]; }

    /**
     * @brief 清空轨迹，删除所有片段
     */
    inline void clear(void) {
        pieces.clear();
        return;
    }

    /**
     * @brief 获取常量迭代器指向第一个片段
     * @return 常量迭代器
     */
    inline typename Pieces::const_iterator begin() const { return pieces.begin(); }

    /**
     * @brief 获取常量迭代器指向最后一个片段之后
     * @return 常量迭代器
     */
    inline typename Pieces::const_iterator end() const { return pieces.end(); }

    /**
     * @brief 获取迭代器指向第一个片段
     * @return 迭代器
     */
    inline typename Pieces::iterator begin() { return pieces.begin(); }

    /**
     * @brief 获取迭代器指向最后一个片段之后
     * @return 迭代器
     */
    inline typename Pieces::iterator end() { return pieces.end(); }

    /**
     * @brief 预留存储空间
     * @param n 预留的片段数量
     */
    inline void reserve(const int& n) {
        pieces.reserve(n > 0 ? static_cast<typename Pieces::size_type>(n) : typename Pieces::size_type{0});
        return;
    }

    /**
     * @brief 在轨迹末尾添加一个片段
     * @param piece 要添加的片段
     */
    inline void emplace_back(const Piece<D>& piece) {
        pieces.emplace_back(piece);
        return;
    }

    /**
     * @brief 在轨迹末尾添加一个片段（通过持续时间和系数矩阵）
     * @param dur 片段持续时间
     * @param cMat 片段系数矩阵
     */
    inline void emplace_back(const double& dur, const typename Piece<D>::CoefficientMat& cMat) {
        pieces.emplace_back(dur, cMat);
        return;
    }

    /**
     * @brief 将另一条轨迹的所有片段追加到当前轨迹末尾
     * @param traj 要追加的轨迹
     */
    inline void append(const Trajectory<D>& traj) {
        pieces.insert(pieces.end(), traj.begin(), traj.end());
        return;
    }

    /**
     * @brief 定位给定时间所在的轨迹片段索引，并将时间转换为该片段的局部时间
     * @param t 输入：全局时间，输出：转换为所在片段的局部时间（引用参数）
     * @return 片段索引
     *
     * 算法：从第一个片段开始，逐个减去每个片段的持续时间，直到找到包含时间t的片段
     * 如果t超出总时长，返回最后一个片段
     */
    inline int locatePieceIdx(double& t) const {
        int N = getPieceNum();
        int idx;
        double dur;
        // 遍历片段，找到时间t所在的片段
        for (idx = 0; idx < N && t > (dur = pieces[static_cast<typename Pieces::size_type>(idx)].getDuration());
             idx++) {
            t -= dur; // 减去当前片段的持续时间
        }
        // 如果时间超出总时长，返回最后一个片段
        if (idx == N) {
            idx--;
            t += pieces[static_cast<typename Pieces::size_type>(idx)].getDuration(); // 恢复到片段末尾的时间
        }
        return idx;
    }

    /**
     * @brief 获取给定时刻的位置
     * @param t 全局时间（从轨迹起点开始）
     * @return 3D位置向量
     *
     * 自动找到时间t所在的片段，然后计算该片段在局部时间的位置
     */
    inline Eigen::Vector3d getPos(double t) const {
        int pieceIdx = locatePieceIdx(t); // t会被修改为局部时间
        return pieces[static_cast<typename Pieces::size_type>(pieceIdx)].getPos(t);
    }

    /**
     * @brief 获取给定时刻的速度
     * @param t 全局时间
     * @return 3D速度向量
     */
    inline Eigen::Vector3d getVel(double t) const {
        int pieceIdx = locatePieceIdx(t);
        return pieces[static_cast<typename Pieces::size_type>(pieceIdx)].getVel(t);
    }

    /**
     * @brief 获取给定时刻的加速度
     * @param t 全局时间
     * @return 3D加速度向量
     */
    inline Eigen::Vector3d getAcc(double t) const {
        int pieceIdx = locatePieceIdx(t);
        return pieces[pieceIdx].getAcc(t);
    }

    /**
     * @brief 获取给定时刻的Jerk
     * @param t 全局时间
     * @return 3D Jerk向量
     */
    inline Eigen::Vector3d getJer(double t) const {
        int pieceIdx = locatePieceIdx(t);
        return pieces[pieceIdx].getJer(t);
    }

    /**
     * @brief 获取指定连接点的位置
     * @param juncIdx 连接点索引（0到N，其中N是片段数量）
     * @return 3D位置向量
     *
     * 连接点是轨迹片段之间的衔接点
     * - juncIdx < N: 返回第juncIdx个片段的起点位置
     * - juncIdx == N: 返回最后一个片段的终点位置
     */
    inline Eigen::Vector3d getJuncPos(int juncIdx) const {
        if (juncIdx != getPieceNum()) {
            return pieces[juncIdx].getCoeffMat().col(D); // 片段起点（t=0时的位置）
        } else {
            return pieces[juncIdx - 1].getPos(pieces[juncIdx - 1].getDuration()); // 最后片段的终点
        }
    }

    /**
     * @brief 获取指定连接点的速度
     * @param juncIdx 连接点索引
     * @return 3D速度向量
     *
     * 返回连接点处的速度（片段起点或终点的速度）
     */
    inline Eigen::Vector3d getJuncVel(int juncIdx) const {
        if (juncIdx != getPieceNum()) {
            return pieces[juncIdx].getCoeffMat().col(D - 1); // 片段起点的速度
        } else {
            return pieces[juncIdx - 1].getVel(pieces[juncIdx - 1].getDuration()); // 最后片段的终点速度
        }
    }

    /**
     * @brief 获取指定连接点的加速度
     * @param juncIdx 连接点索引
     * @return 3D加速度向量
     *
     * 返回连接点处的加速度（片段起点或终点的加速度）
     */
    inline Eigen::Vector3d getJuncAcc(int juncIdx) const {
        if (juncIdx != getPieceNum()) {
            return pieces[juncIdx].getCoeffMat().col(D - 2) * 2.0; // 片段起点的加速度
        } else {
            return pieces[juncIdx - 1].getAcc(pieces[juncIdx - 1].getDuration()); // 最后片段的终点加速度
        }
    }

    /**
     * @brief 获取整条轨迹的最大速度
     * @return 最大速度的模长
     *
     * 遍历所有片段，找到最大速度
     */
    inline double getMaxVelRate() const {
        int N = getPieceNum();
        double maxVelRate = -INFINITY;
        double tempNorm;
        for (int i = 0; i < N; i++) {
            tempNorm = pieces[i].getMaxVelRate(); // 获取每个片段的最大速度
            maxVelRate = maxVelRate < tempNorm ? tempNorm : maxVelRate;
        }
        return maxVelRate;
    }

    /**
     * @brief 获取整条轨迹的最大加速度
     * @return 最大加速度的模长
     *
     * 遍历所有片段，找到最大加速度
     */
    inline double getMaxAccRate() const {
        int N = getPieceNum();
        double maxAccRate = -INFINITY;
        double tempNorm;
        for (int i = 0; i < N; i++) {
            tempNorm = pieces[i].getMaxAccRate(); // 获取每个片段的最大加速度
            maxAccRate = maxAccRate < tempNorm ? tempNorm : maxAccRate;
        }
        return maxAccRate;
    }

    /**
     * @brief 检查整条轨迹是否满足最大速度约束
     * @param maxVelRate 最大允许速度
     * @return 如果所有片段都满足约束返回true，否则返回false
     *
     * 遍历所有片段，检查每个片段是否都满足速度约束
     */
    inline bool checkMaxVelRate(const double& maxVelRate) const {
        int N = getPieceNum();
        bool feasible = true;
        for (int i = 0; i < N && feasible; i++) {
            feasible = feasible && pieces[i].checkMaxVelRate(maxVelRate);
        }
        return feasible;
    }

    /**
     * @brief 检查整条轨迹是否满足最大加速度约束
     * @param maxAccRate 最大允许加速度
     * @return 如果所有片段都满足约束返回true，否则返回false
     *
     * 遍历所有片段，检查每个片段是否都满足加速度约束
     */
    inline bool checkMaxAccRate(const double& maxAccRate) const {
        int N = getPieceNum();
        bool feasible = true;
        for (int i = 0; i < N && feasible; i++) {
            feasible = feasible && pieces[i].checkMaxAccRate(maxAccRate);
        }
        return feasible;
    }
};

} // namespace xgc2_math::optimization
#endif
// NOLINTEND
