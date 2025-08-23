#include "../include/geometry/Planimetry/Parallelogram.h"

using namespace SharedMath::Geometry;

bool Parallelogram::isParallelogram(const std::array<Point2D, 4>& vertices){
    // Проверка на уникальность точек
    for(auto i = 0; i < vertices.size(); ++i){
        for(auto j = i + 1; j < vertices.size(); j++){
            if(vertices[i] == vertices[j]) return false;
        }
    }
    Vector2D v1(vertices[0], vertices[1]);
    Vector2D v2(vertices[1], vertices[2]);
    Vector2D v3(vertices[2], vertices[3]);
    Vector2D v4(vertices[3], vertices[0]);

    bool sidesParallel = v1.isParallel(v3) && v2.isParallel(v4);
    bool sidesEqualLength = std::abs(v1.length() - v3.length()) < Epsilon && 
                           std::abs(v2.length() - v4.length()) < Epsilon;

    // Диагонали должны делиться пополам
    Point2D diagonal1_midpoint((vertices[0].x() + vertices[2].x()) / 2.0,
                              (vertices[0].y() + vertices[2].y()) / 2.0);
    Point2D diagonal2_midpoint((vertices[1].x() + vertices[3].x()) / 2.0,
                              (vertices[1].y() + vertices[3].y()) / 2.0);

    bool diagonalsBisect = std::abs(diagonal1_midpoint.x() - diagonal2_midpoint.x()) < Epsilon &&
                          std::abs(diagonal1_midpoint.y() - diagonal2_midpoint.y()) < Epsilon;

    return sidesParallel && sidesEqualLength && diagonalsBisect;
}

Point2D Parallelogram::findDownLeftPoint(const std::array<Point2D, 4>& points){
    Point2D result = points[0];
    
    for(auto i = 1; i < points.size(); ++i){
        if(points[i].x() < result.x() || 
           (std::abs(points[i].x() - result.x()) < Epsilon && points[i].y() < result.y())){
            result = points[i];
        }
    }
    return result;
}

Point2D Parallelogram::findDownRightPoint(const std::array<Point2D, 4>& points){
    Point2D result = points[0];
    
    for(auto i = 1; i < points.size(); ++i){
        if(points[i].y() < result.y() || 
           (std::abs(points[i].y() - result.y()) < Epsilon && points[i].x() > result.x())){
            result = points[i];
        }
    }
    return result;
}

Point2D Parallelogram::findUpLeftPoint(const std::array<Point2D, 4>& points){
    Point2D result = points[0];
    
    for(auto i = 1; i < points.size(); ++i){
        if(points[i].x() < result.x() || 
           (std::abs(points[i].x() - result.x()) < Epsilon && points[i].y() > result.y())){
            result = points[i];
        }
    }
    return result;
}

Point2D Parallelogram::findUpRightPoint(const std::array<Point2D, 4>& points){
    Point2D result = points[0];
    
    for(auto i = 1; i < points.size(); ++i){
        if(points[i].x() > result.x() || 
           (std::abs(points[i].x() - result.x()) < Epsilon && points[i].y() > result.y())){
            result = points[i];
        }
    }
    return result;
}


Parallelogram::Parallelogram(const std::array<Point2D, 4>& points){
    if(!isParallelogram(points)) {
        throw std::invalid_argument("Points do not form a parallelogram");
    }
    std::array<Point2D, 4> orderedPoints;
    
    Point2D downLeft = findDownLeftPoint(points);
    orderedPoints[0] = downLeft;
    
    std::vector<Point2D> remainingPoints;
    for(const auto& point : points) {
        if(point != downLeft) {
            remainingPoints.push_back(point);
        }
    }
    
    std::sort(remainingPoints.begin(), remainingPoints.end(), 
        [&downLeft](const Point2D& a, const Point2D& b) {
            Vector2D va(downLeft, a);
            Vector2D vb(downLeft, b);
            return std::atan2(va.y(), va.x()) < std::atan2(vb.y(), vb.x());
        });
    
    orderedPoints[1] = remainingPoints[0];
    orderedPoints[2] = remainingPoints[2];
    orderedPoints[3] = remainingPoints[1];
    
    setVertices(orderedPoints);
}

double Parallelogram::area() const {
    const auto& vertices = getVertices();
    
    Vector2D v1(vertices[0], vertices[1]);
    Vector2D v2(vertices[0], vertices[3]);
    
    double crossProduct = v1.cross(v2);
    return std::abs(crossProduct);
}

double Parallelogram::perimeter() const {
    const auto& vertices = getVertices();
    
    double side1 = Vector2D(vertices[0], vertices[1]).length();
    double side2 = Vector2D(vertices[1], vertices[2]).length();
    
    return 2.0 * (side1 + side2);
}