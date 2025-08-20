#pragma once
#include "../constans.h"

namespace SharedMath{
    namespace Geometry{
        class Angle{
        public:
            enum TYPE{
                RADIANS,
                DEGREES
            };

            Angle(double angle, TYPE type){
                if(type == RADIANS){
                    radians = angle;
                    degrees = angle * 180.0 / Pi;
                }else{
                    degrees = angle;
                    radians = angle * Pi / 180.0;
                }
            }

            double getRadians() const{return radians;}
            double getDegrees() const{return degrees;}

            Angle() = default;
            Angle(const Angle&) = default;
            Angle(Angle&&) = default;
            Angle& operator=(const Angle&) = default;
            Angle& operator=(Angle&&) = default;
            ~Angle() = default;

        private:
            double degrees;
            double radians;
        };
    }
}