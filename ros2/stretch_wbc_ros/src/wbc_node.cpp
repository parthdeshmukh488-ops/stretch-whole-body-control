// stretch_wbc node: reach a tf2 target frame with the whole body, through the safety filter.
//
// Each control tick it looks up the target frame in the world frame through tf2,
// takes one Jacobian-transpose step towards it, turns that step into a joint
// velocity, passes the velocity through the safety filter and integrates it.
// The robot is simulated kinematically inside the node: it publishes the joint
// state and the base and tool frames, so RViz shows it moving. Driving a real
// Stretch would replace the integration with Hello Robot's ROS 2 interfaces.
//
//   ros2 launch stretch_wbc_ros wbc.launch.py x:=0.6 y:=0.4 z:=0.8

#include <chrono>
#include <memory>
#include <string>

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "stretch_wbc_ros/controller.hpp"
#include "tf2/exceptions.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_ros/transform_listener.h"

namespace stretch_wbc_ros {

class WbcNode : public rclcpp::Node {
 public:
  WbcNode() : Node("stretch_wbc"), limits_(Limits::stretch3()), filter_(limits_.velocity, 1) {
    world_frame_ = declare_parameter<std::string>("world_frame", "odom");
    target_frame_ = declare_parameter<std::string>("target_frame", "target");
    tolerance_ = declare_parameter<double>("tolerance_m", 0.002);
    dt_ = 1.0 / declare_parameter<double>("rate_hz", 50.0);

    q_.setZero();
    q_(kLift) = 0.6;  // start mid-height, arm retracted, wrist straight

    buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    listener_ = std::make_shared<tf2_ros::TransformListener>(*buffer_);
    broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    joint_publisher_ = create_publisher<sensor_msgs::msg::JointState>("joint_states", 10);
    timer_ = create_wall_timer(std::chrono::duration<double>(dt_), [this] { tick(); });
  }

 private:
  void tick() {
    JointVector velocity = JointVector::Zero();
    try {
      const auto t = buffer_->lookupTransform(world_frame_, target_frame_, tf2::TimePointZero);
      const Eigen::Vector3d target(t.transform.translation.x, t.transform.translation.y,
                                   t.transform.translation.z);
      if ((target - kinematics_.position(q_)).norm() > tolerance_) {
        const JointVector step = jacobian_transpose_step(kinematics_, q_, target);
        configure_filter(filter_, limits_, q_);
        velocity = filter_.solve(step / dt_).dq;
      }
    } catch (const tf2::TransformException& ex) {
      // No target: hold still rather than act on stale information.
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "no target frame: %s", ex.what());
    }
    q_ = limits_.clamp(q_ + velocity * dt_);
    publish(velocity);
  }

  void publish(const JointVector& velocity) {
    const auto now = get_clock()->now();
    sensor_msgs::msg::JointState joints;
    joints.header.stamp = now;
    joints.name = {"base_x", "base_y", "base_theta", "lift",
                   "arm",    "wrist_yaw", "wrist_pitch", "wrist_roll"};
    joints.position.assign(q_.data(), q_.data() + kNumJoints);
    joints.velocity.assign(velocity.data(), velocity.data() + kNumJoints);
    joint_publisher_->publish(joints);

    const Eigen::Quaterniond base_rotation(
        Eigen::AngleAxisd(q_(kBaseTheta), Eigen::Vector3d::UnitZ()));
    const Eigen::Quaterniond tool_rotation(kinematics_.orientation(q_));
    const Eigen::Vector3d base_position(q_(kBaseX), q_(kBaseY), 0.0);
    broadcaster_->sendTransform({frame(now, "base_link", base_position, base_rotation),
                                 frame(now, "tool", kinematics_.position(q_), tool_rotation)});
  }

  geometry_msgs::msg::TransformStamped frame(const rclcpp::Time& stamp, const std::string& child,
                                             const Eigen::Vector3d& p,
                                             const Eigen::Quaterniond& r) const {
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = stamp;
    t.header.frame_id = world_frame_;
    t.child_frame_id = child;
    t.transform.translation.x = p.x();
    t.transform.translation.y = p.y();
    t.transform.translation.z = p.z();
    t.transform.rotation.x = r.x();
    t.transform.rotation.y = r.y();
    t.transform.rotation.z = r.z();
    t.transform.rotation.w = r.w();
    return t;
  }

  Limits limits_;
  StretchKinematics kinematics_;
  stretchwbc::SafetyFilter filter_;
  JointVector q_;
  std::string world_frame_;
  std::string target_frame_;
  double tolerance_{0.002};
  double dt_{0.02};

  std::unique_ptr<tf2_ros::Buffer> buffer_;
  std::shared_ptr<tf2_ros::TransformListener> listener_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> broadcaster_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace stretch_wbc_ros

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<stretch_wbc_ros::WbcNode>());
  rclcpp::shutdown();
  return 0;
}
