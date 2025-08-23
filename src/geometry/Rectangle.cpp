#include "../include/geometry/Rectangle.h"
#include <algorithm>
#include <cmath>

using namespace SharedMath::Geometry;

Rectangle::Rectangle(const std::array<Point2D, 4>& points) {
    if (!isRectangle(points)) {
        throw std::invalid_argument("Points do not form a rectangle");
    }
    setVertices(orderRectanglePoints(points));
}

Rectangle::Rectangle(const Point2D& bottomLeft, const Point2D& topRight) {
    double x1 = bottomLeft.x();
    double y1 = bottomLeft.y();
    double x2 = topRight.x();
    double y2 = topRight.y();

    std::array<Point2D, 4> points = {
        Point2D(x1, y1),
        Point2D(x2, y1),
        Point2D(x2, y2),
        Point2D(x1, y2)
    };
    
    setVertices(points);
}

Rectangle::Rectangle(const Point2D& position, double width, double height) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("Width and height must be positive");
    }

    std::array<Point2D, 4> points = {
        position,
        Point2D(position.x() + width, position.y()),
        Point2D(position.x() + width, position.y() + height),
        Point2D(position.x(), position.y() + height)
    };
    
    setVertices(points);
}

bool Rectangle::isRectangle(const std::array<Point2D, 4>& vertices) {
    if (!Parallelogram::isParallelogram(vertices)) {
        return false;
    }

    Vector2D v1(vertices[0], vertices[1]);
    Vector2D v2(vertices[0], vertices[3]);

    return std::abs(v1.dot(v2)) < Epsilon;
}

double Rectangle::getWidth() const {
    const auto& vertices = getVertices();
    return Vector2D(vertices[0], vertices[1]).length();
}

double Rectangle::getHeight() const {
    const auto& vertices = getVertices();
    return Vector2D(vertices[0], vertices[3]).length();
}

double Rectangle::getAspectRatio() const {
    double width = getWidth();
    double height = getHeight();
    return (height > Epsilon) ? width / height : std::numeric_limits<double>::infinity();
}

Point2D Rectangle::getCenter() const {
    const auto& vertices = getVertices();
    return Point2D(
        (vertices[0].x() + vertices[2].x()) / 2.0,
        (vertices[0].y() + vertices[2].y()) / 2.0
    );
}

Point2D Rectangle::getBottomLeft() const {
    return getVertices()[0];
}

Point2D Rectangle::getBottomRight() const {
    return getVertices()[1];
}

Point2D Rectangle::getTopLeft() const {
    return getVertices()[3];
}

Point2D Rectangle::getTopRight() const {
    return getVertices()[2];
}

void Rectangle::setSize(double width, double height) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("Width and height must be positive");
    }

    Point2D bottomLeft = getBottomLeft();
    std::array<Point2D, 4> points = {
        bottomLeft,
        Point2D(bottomLeft.x() + width, bottomLeft.y()),
        Point2D(bottomLeft.x() + width, bottomLeft.y() + height),
        Point2D(bottomLeft.x(), bottomLeft.y() + height)
    };
    
    setVertices(points);
}

void Rectangle::setPosition(const Point2D& position) {
    double width = getWidth();
    double height = getHeight();
    
    std::array<Point2D, 4> points = {
        position,
        Point2D(position.x() + width, position.y()),
        Point2D(position.x() + width, position.y() + height),
        Point2D(position.x(), position.y() + height)
    };
    
    setVertices(points);
}

void Rectangle::move(const Vector2D& offset) {
    auto vertices = getVertices();
    for (auto& vertex : vertices) {
        vertex = Point2D(vertex.x() + offset.x(), vertex.y() + offset.y());
    }
    setVertices(vertices);
}

void Rectangle::scale(double factor) {
    if (factor <= Epsilon) {
        throw std::invalid_argument("Scale factor must be positive");
    }
    
    Point2D center = getCenter();
    auto vertices = getVertices();
    
    for (auto& vertex : vertices) {
        Vector2D vec(center, vertex);
        vertex = Point2D(center.x() + vec.x() * factor, 
                         center.y() + vec.y() * factor);
    }
    
    setVertices(vertices);
}

void Rectangle::scale(double widthFactor, double heightFactor) {
    if (widthFactor <= Epsilon || heightFactor <= Epsilon) {
        throw std::invalid_argument("Scale factors must be positive");
    }
    
    Point2D center = getCenter();
    auto vertices = getVertices();
    
    for (auto& vertex : vertices) {
        double dx = (vertex.x() - center.x()) * widthFactor;
        double dy = (vertex.y() - center.y()) * heightFactor;
        vertex = Point2D(center.x() + dx, center.y() + dy);
    }
    
    setVertices(vertices);
}

bool Rectangle::isSquare() const {
    double width = getWidth();
    double height = getHeight();
    return std::abs(width - height) < Epsilon;
}

bool Rectangle::contains(const Point2D& point) const {
    const auto& vertices = getVertices();
    double minX = std::min({vertices[0].x(), vertices[1].x(), vertices[2].x(), vertices[3].x()});
    double maxX = std::max({vertices[0].x(), vertices[1].x(), vertices[2].x(), vertices[3].x()});
    double minY = std::min({vertices[0].y(), vertices[1].y(), vertices[2].y(), vertices[3].y()});
    double maxY = std::max({vertices[0].y(), vertices[1].y(), vertices[2].y(), vertices[3].y()});
    
    return point.x() >= minX && point.x() <= maxX && 
           point.y() >= minY && point.y() <= maxY;
}

bool Rectangle::intersects(const Rectangle& other) const {
    const auto& thisVertices = getVertices();
    const auto& otherVertices = other.getVertices();
    
    double thisMinX = std::min({thisVertices[0].x(), thisVertices[1].x(), thisVertices[2].x(), thisVertices[3].x()});
    double thisMaxX = std::max({thisVertices[0].x(), thisVertices[1].x(), thisVertices[2].x(), thisVertices[3].x()});
    double thisMinY = std::min({thisVertices[0].y(), thisVertices[1].y(), thisVertices[2].y(), thisVertices[3].y()});
    double thisMaxY = std::max({thisVertices[0].y(), thisVertices[1].y(), thisVertices[2].y(), thisVertices[3].y()});
    
    double otherMinX = std::min({otherVertices[0].x(), otherVertices[1].x(), otherVertices[2].x(), otherVertices[3].x()});
    double otherMaxX = std::max({otherVertices[0].x(), otherVertices[1].x(), otherVertices[2].x(), otherVertices[3].x()});
    double otherMinY = std::min({otherVertices[0].y(), otherVertices[1].y(), otherVertices[2].y(), otherVertices[3].y()});
    double otherMaxY = std::max({otherVertices[0].y(), otherVertices[1].y(), otherVertices[2].y(), otherVertices[3].y()});
    
    return !(thisMaxX < otherMinX || thisMinX > otherMaxX ||
             thisMaxY < otherMinY || thisMinY > otherMaxY);
}

bool Rectangle::operator==(const Rectangle& other) const {
    const auto& thisVertices = getVertices();
    const auto& otherVertices = other.getVertices();
    
    for (size_t i = 0; i < 4; ++i) {
        if (thisVertices[i] != otherVertices[i]) {
            return false;
        }
    }
    return true;
}

bool Rectangle::operator!=(const Rectangle& other) const {
    return !(*this == other);
}

Rectangle Rectangle::operator+(const Vector2D& offset) const {
    Rectangle result = *this;
    result.move(offset);
    return result;
}

Rectangle Rectangle::operator-(const Vector2D& offset) const {
    Rectangle result = *this;
    result.move(Vector2D(-offset.x(), -offset.y()));
    return result;
}

Rectangle& Rectangle::operator+=(const Vector2D& offset) {
    move(offset);
    return *this;
}

Rectangle& Rectangle::operator-=(const Vector2D& offset) {
    move(Vector2D(-offset.x(), -offset.y()));
    return *this;
}

std::array<Point2D, 4> Rectangle::orderRectanglePoints(const std::array<Point2D, 4>& points) {
    double minX = points[0].x();
    double maxX = points[0].x();
    double minY = points[0].y();
    double maxY = points[0].y();
    
    for (const auto& point : points) {
        minX = std::min(minX, point.x());
        maxX = std::max(maxX, point.x());
        minY = std::min(minY, point.y());
        maxY = std::max(maxY, point.y());
    }
    
    return {
        Point2D(minX, minY), // bottom-left
        Point2D(maxX, minY), // bottom-right
        Point2D(maxX, maxY), // top-right
        Point2D(minX, maxY)  // top-left
    };
}