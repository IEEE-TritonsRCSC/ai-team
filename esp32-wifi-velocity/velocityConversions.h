#include<cmath>
#include<array>
#include<iostream>
#include<climits>

// wheel constants
constexpr double WHEEL_RADIUS = 0.02425;
constexpr double GEAR_RATIO = 36;
constexpr double MAX_RPM = 15000;
constexpr double MAX_VELOCITY = 175;
constexpr double RESCALE_FACTOR = MAX_RPM / MAX_VELOCITY;

// robot geometry
constexpr double ROBOT_RADIUS = 0.09;           // distance from robot center to wheel contact point
constexpr double DEG2RAD = M_PI / 180.0;

// motor positions (given): FR=30°, FL=150°, BL=240°, BR=300°
constexpr double FR_MOTOR_ANGLE = 30.0 * DEG2RAD;
constexpr double FL_MOTOR_ANGLE = 150.0 * DEG2RAD;
constexpr double BL_MOTOR_ANGLE = 240.0 * DEG2RAD;
constexpr double BR_MOTOR_ANGLE = 300.0 * DEG2RAD;

// wheel drive direction is tangent to the circle.
// Set TANGENT_SIGN = -1 for clockwise tangent (angle = motor - 90°),
// or +1 for counterclockwise tangent (angle = motor + 90°).
constexpr int TANGENT_SIGN = -1;

// wheel positions (Xi, Yi) from motor angles
constexpr double FR_X = ROBOT_RADIUS * cos(FR_MOTOR_ANGLE);
constexpr double FR_Y = ROBOT_RADIUS * sin(FR_MOTOR_ANGLE);
constexpr double FL_X = ROBOT_RADIUS * cos(FL_MOTOR_ANGLE);
constexpr double FL_Y = ROBOT_RADIUS * sin(FL_MOTOR_ANGLE);
constexpr double BL_X = ROBOT_RADIUS * cos(BL_MOTOR_ANGLE);
constexpr double BL_Y = ROBOT_RADIUS * sin(BL_MOTOR_ANGLE);
constexpr double BR_X = ROBOT_RADIUS * cos(BR_MOTOR_ANGLE);
constexpr double BR_Y = ROBOT_RADIUS * sin(BR_MOTOR_ANGLE);

// wheel drive angles phi_i (direction along which each wheel provides traction)
constexpr double FR_WHEEL_ANGLE = FR_MOTOR_ANGLE + TANGENT_SIGN * (M_PI / 2.0);
constexpr double FL_WHEEL_ANGLE = FL_MOTOR_ANGLE + TANGENT_SIGN * (M_PI / 2.0);
constexpr double BL_WHEEL_ANGLE = BL_MOTOR_ANGLE + TANGENT_SIGN * (M_PI / 2.0);
constexpr double BR_WHEEL_ANGLE = BR_MOTOR_ANGLE + TANGENT_SIGN * (M_PI / 2.0);

// API
void getVelocityArray(std::array<int, 4>& wheel_speeds, double heading, double vx,
                      double vy, double rotV);

void valuesToBytes(std::array<int, 4>& wheel_speeds, std::array<unsigned char, 8>& wheel_speeds_byte);

void action_to_byte_array(std::array<unsigned char, 8>& wheel_speeds_byte);

void action_to_byte_array_with_params(std::array<unsigned char, 8>& wheel_speeds_byte,
                                      double vx, double vy, double rotV);