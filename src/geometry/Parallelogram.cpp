#include "../include/geometry/Parallelogram.h"

using namespace SharedMath::Geometry;

bool Parallelogram::isParallelogram(const std::array<Point<2>, 4>& vertices){
    for(auto i = 0; i < vertices.size(); ++i){
        for(auto j = i + 1; j < vertices.size(); j++){
            if(vertices[i] == vertices[j])return false;
        }
    }


}

Point<2>& Parallelogram::findDownLeftPoint(const std::array<Point<2>, 4>& points) const{
    double minX = points[0][0];
    double minY = points[0][1];

    

}

Parallelogram::Parallelogram(const std::array<Point<2>, 4>& points){

}

double Parallelogram::area() const{
    
} 