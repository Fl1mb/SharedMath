#include "../include/geometry/Planimetry/Triangle.h"

using namespace SharedMath::Geometry;

Triangle::Triangle(const std::array<Point2D, 3>& points) : Polygon<3>(points){
    if(!isValidTriangle(points)){
        throw std::invalid_argument("Invalid points for triangle");
    }
}

Triangle::Triangle(const Point2D& a, const Point2D& b, const Point2D& c){
    std::array<Point2D, 3> vertices {a, b, c};
    if(!isValidTriangle(vertices)){
        throw std::invalid_argument("Invalid points for triangle");
    }
    setVertices(vertices);
}


bool Triangle::isValidTriangle(const std::array<Point2D, 3>& vertices){
    if(arePointsCollinear(vertices[0], vertices[1], vertices[2])){
        return false;
    }

    for(size_t i = 0; i < 3; ++i){
        for(size_t j = i + 1; j < 3; ++j){
            if(vertices[i] == vertices[j])return false;
        }
    }
    return true;
}

bool Triangle::arePointsCollinear(const Point2D& a, const Point2D& b, const Point2D& c){
    double area = std::abs((b.x() - a.x()) * (c.y() - a.y()) - 
                          (b.y() - a.y()) * (c.x() - a.x()));
    return area < Epsilon;
}

double Triangle::area() const{
    const auto& vertices = getVertices();
    return calculateTriangleArea(vertices[0], vertices[1], vertices[2]);
}

double Triangle::calculateTriangleArea(const Point2D& a, const Point2D& b, const Point2D& c){
    return std::abs((b.x() - a.x()) * (c.y() - a.y()) - 
                   (b.y() - a.y()) * (c.x() - a.x())) / 2.0;
}

double Triangle::perimeter() const{
    const auto& vertices = getVertices();
    double side1 = Vector2D(vertices[0], vertices[1]).length();
    double side2 = Vector2D(vertices[1], vertices[2]).length();
    double side3 = Vector2D(vertices[2], vertices[0]).length();
    return side1 + side2 + side3;
}

double Triangle::getSideLength(size_t sideIndex) const {
    if (sideIndex >= 3) {
        throw std::out_of_range("Side index must be 0, 1, or 2");
    }

    const auto& vertices = getVertices();
    size_t nextIndex = (sideIndex + 1) % 3;
    return Vector2D(vertices[sideIndex], vertices[nextIndex]).length();
}

double Triangle::getAngle(size_t vertexIndex) const {
    if (vertexIndex >= 3) {
        throw std::out_of_range("Vertex index must be 0, 1, or 2");
    }

    const auto& vertices = getVertices();
    size_t prevIndex = (vertexIndex + 2) % 3;
    size_t nextIndex = (vertexIndex + 1) % 3;

    Vector2D v1(vertices[vertexIndex], vertices[prevIndex]);
    Vector2D v2(vertices[vertexIndex], vertices[nextIndex]);

    double dotProduct = v1.dot(v2);
    double lengths = v1.length() * v2.length();

    if (lengths < Epsilon) {
        return 0.0;
    }

    double cosAngle = dotProduct / lengths;
    cosAngle = std::max(-1.0, std::min(1.0, cosAngle));
    return std::acos(cosAngle);
}

double Triangle::getAltitude(size_t sideIndex) const {
    double area = this->area();
    double baseLength = getSideLength(sideIndex);
    return (baseLength > Epsilon) ? (2.0 * area) / baseLength : 0.0;
}

Point2D Triangle::getCentroid() const {
    const auto& vertices = getVertices();
    return Point2D(
        (vertices[0].x() + vertices[1].x() + vertices[2].x()) / 3.0,
        (vertices[0].y() + vertices[1].y() + vertices[2].y()) / 3.0
    );
}

Point2D Triangle::getCircumcenter() const {
    const auto& vertices = getVertices();
    const Point2D& a = vertices[0];
    const Point2D& b = vertices[1];
    const Point2D& c = vertices[2];

    double d = 2 * (a.x() * (b.y() - c.y()) + b.x() * (c.y() - a.y()) + c.x() * (a.y() - b.y()));
    
    if (std::abs(d) < Epsilon) {
        throw std::runtime_error("Points are collinear, no circumcenter");
    }

    double ux = ((a.x() * a.x() + a.y() * a.y()) * (b.y() - c.y()) +
                (b.x() * b.x() + b.y() * b.y()) * (c.y() - a.y()) +
                (c.x() * c.x() + c.y() * c.y()) * (a.y() - b.y())) / d;

    double uy = ((a.x() * a.x() + a.y() * a.y()) * (c.x() - b.x()) +
                (b.x() * b.x() + b.y() * b.y()) * (a.x() - c.x()) +
                (c.x() * c.x() + c.y() * c.y()) * (b.x() - a.x())) / d;

    return Point2D(ux, uy);
}

Point2D Triangle::getIncenter() const {
    const auto& vertices = getVertices();
    const Point2D& a = vertices[0];
    const Point2D& b = vertices[1];
    const Point2D& c = vertices[2];

    double sideA = Vector2D(b, c).length();
    double sideB = Vector2D(a, c).length();
    double sideC = Vector2D(a, b).length();
    double perimeter = sideA + sideB + sideC;

    if (perimeter < Epsilon) {
        throw std::runtime_error("Degenerate triangle");
    }

    double x = (sideA * a.x() + sideB * b.x() + sideC * c.x()) / perimeter;
    double y = (sideA * a.y() + sideB * b.y() + sideC * c.y()) / perimeter;

    return Point2D(x, y);
}

bool Triangle::isEquilateral() const {
    double side1 = getSideLength(0);
    double side2 = getSideLength(1);
    double side3 = getSideLength(2);

    return std::abs(side1 - side2) < Epsilon &&
           std::abs(side2 - side3) < Epsilon &&
           std::abs(side3 - side1) < Epsilon;
}

bool Triangle::isIsosceles() const {
    double side1 = getSideLength(0);
    double side2 = getSideLength(1);
    double side3 = getSideLength(2);

    return std::abs(side1 - side2) < Epsilon ||
           std::abs(side2 - side3) < Epsilon ||
           std::abs(side3 - side1) < Epsilon;
}

bool Triangle::isRight() const {
    double side1 = getSideLength(0);
    double side2 = getSideLength(1);
    double side3 = getSideLength(2);

    // Проверка теоремы Пифагора
    double sides[3] = {side1, side2, side3};
    std::sort(sides, sides + 3);

    return std::abs(sides[2] * sides[2] - (sides[0] * sides[0] + sides[1] * sides[1])) < Epsilon;
}

bool Triangle::isAcute() const {
    const auto& vertices = getVertices();
    for (size_t i = 0; i < 3; ++i) {
        double angle = getAngle(i);
        if (angle >= M_PI_2 - Epsilon) {
            return false;
        }
    }
    return true;
}

bool Triangle::isObtuse() const {
    const auto& vertices = getVertices();
    for (size_t i = 0; i < 3; ++i) {
        double angle = getAngle(i);
        if (angle > M_PI_2 + Epsilon) {
            return true;
        }
    }
    return false;
}

bool Triangle::contains(const Point2D& point) const {
    return isPointInside(point);
}

bool Triangle::isPointInside(const Point2D& point) const {
    const auto& vertices = getVertices();
    
    // Метод барицентрических координат
    double areaABC = area();
    double areaPBC = calculateTriangleArea(point, vertices[1], vertices[2]);
    double areaPCA = calculateTriangleArea(point, vertices[2], vertices[0]);
    double areaPAB = calculateTriangleArea(point, vertices[0], vertices[1]);

    double totalArea = areaPBC + areaPCA + areaPAB;
    return std::abs(totalArea - areaABC) < Epsilon;
}

void Triangle::move(const Vector2D& offset) {
    auto vertices = getVertices();
    for (auto& vertex : vertices) {
        vertex = Point2D(vertex.x() + offset.x(), vertex.y() + offset.y());
    }
    setVertices(vertices);
}

void Triangle::scale(double factor) {
    if (factor <= Epsilon) {
        throw std::invalid_argument("Scale factor must be positive");
    }

    auto vertices = getVertices();
    
    for (auto& vertex : vertices) {
        vertex = Point2D(vertex.x() * factor, vertex.y() * factor);
    }
    
    setVertices(vertices);
}
void Triangle::rotate(double angle, const Point2D& center) {
    double cosA = std::cos(angle);
    double sinA = std::sin(angle);
    
    auto vertices = getVertices();
    
    for (auto& vertex : vertices) {
        double dx = vertex.x() - center.x();
        double dy = vertex.y() - center.y();
        
        vertex = Point2D(
            center.x() + dx * cosA - dy * sinA,
            center.y() + dx * sinA + dy * cosA
        );
    }
    
    setVertices(vertices);
}

bool Triangle::operator==(const Triangle& other) const {
    const auto& thisVertices = getVertices();
    const auto& otherVertices = other.getVertices();
    
    for (size_t i = 0; i < 3; ++i) {
        if (thisVertices[i] != otherVertices[i]) {
            return false;
        }
    }
    return true;
}

bool Triangle::operator!=(const Triangle& other) const {
    return !(*this == other);
}

Triangle Triangle::operator+(const Vector2D& offset) const {
    Triangle result = *this;
    result.move(offset);
    return result;
}

Triangle Triangle::operator-(const Vector2D& offset) const {
    Triangle result = *this;
    result.move(Vector2D(-offset.x(), -offset.y()));
    return result;
}

Triangle& Triangle::operator+=(const Vector2D& offset) {
    move(offset);
    return *this;
}

Triangle& Triangle::operator-=(const Vector2D& offset) {
    move(Vector2D(-offset.x(), -offset.y()));
    return *this;
}

