#include "drake/systems/primitives/sine.h"
#include "drake/systems/primitives/gain.h"

#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/framework/leaf_system.h"

#include "drake/systems/lcm/lcm_publisher_system.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/lcmt_drake_signal.hpp"
#include "drake/systems/lcm/lcmt_drake_signal_translator.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/analysis/explicit_euler_integrator.h"

namespace drake {
namespace systems {

void DoMain() {
  systems::DiagramBuilder<double> builder;

  // Create the const vector source.
  Eigen::VectorXd const_vec = Eigen::VectorXd::Zero(1);
  const_vec(0) = 3.33;
  ConstantVectorSource<double>* const_src =
      builder.AddSystem<ConstantVectorSource<double>>(const_vec);

  // create the lcm publisher
  drake::lcm::DrakeLcm lcm;

  systems::lcm::LcmtDrakeSignalTranslator const_translator(1);
  auto sys_pub = builder.AddSystem(
      std::make_unique<systems::lcm::LcmPublisherSystem>(
          "CONST_STATUS", const_translator, &lcm));
  sys_pub->set_publish_period(0.01);
  // Connect constant source to lcm publisher
  builder.Connect(const_src->get_output_port(), sys_pub->get_input_port());

  // Create a gain source
  Gain<double>* gain_src =
      builder.AddSystem<Gain<double>>(2, 1);
  systems::lcm::LcmtDrakeSignalTranslator gain_translator(1);
  auto gain_pub = builder.AddSystem(
      std::make_unique<systems::lcm::LcmPublisherSystem>(
          "GAIN_STATUS", gain_translator, &lcm));
  gain_pub->set_publish_period(0.01);
  builder.Connect(const_src->get_output_port(), gain_src->get_input_port());
  builder.Connect(gain_src->get_output_port(), gain_pub->get_input_port());

  // Create the sine source.
//  Sine<double>* sine_src =
//      builder.AddSystem<Sine<double>>(1, 1, 1, 2);
//  Sine<double>* sine_src =
//      builder.AddSystem<Sine<double>>(Eigen::Vector2d(1.5, 3.5),  // Amp
//                                      Eigen::Vector2d(1.0, M_PI),  // Freq
//                                      Eigen::Vector2d(0.0, M_PI / 2.0), true); // Phase
  Sine<double>* sine_src =
      builder.AddSystem<Sine<double>>(Eigen::Vector2d(1, 1),  // Amp
                                      Eigen::Vector2d(1, 1),  // Freq
                                      Eigen::Vector2d(0, 0), true); // Phase
  systems::lcm::LcmtDrakeSignalTranslator sine_translator(2);
  auto sine_pub_pos = builder.AddSystem(
      std::make_unique<systems::lcm::LcmPublisherSystem>(
          "SINE_POS", sine_translator, &lcm));
  sine_pub_pos->set_publish_period(0.01);

  auto sine_pub_vel = builder.AddSystem(
      std::make_unique<systems::lcm::LcmPublisherSystem>(
          "SINE_VEL", sine_translator, &lcm));
  sine_pub_pos->set_publish_period(0.01);

  auto sine_pub_accel = builder.AddSystem(
      std::make_unique<systems::lcm::LcmPublisherSystem>(
          "SINE_ACCEL", sine_translator, &lcm));
  sine_pub_pos->set_publish_period(0.01);

  // Create a dummy input to make the simulation work.
//  Eigen::VectorXd const_vec2 = Eigen::VectorXd::Zero(2);
//  const_vec2(0) = 3.33;
//  const_vec2(1) = 4;
//  ConstantVectorSource<double>* const_src2 =
//      builder.AddSystem<ConstantVectorSource<double>>(const_vec2);
//  builder.Connect(const_src2->get_output_port(), sine_src->get_input_port());

  // Connect the sine output to the publisher.
  builder.Connect(sine_src->get_output_port(0), sine_pub_pos->get_input_port());
  builder.Connect(sine_src->get_output_port(1), sine_pub_vel->get_input_port());
  builder.Connect(sine_src->get_output_port(2), sine_pub_accel->get_input_port());

  // Build the diagram.
  std::unique_ptr<systems::Diagram<double>> diagram = builder.Build();
  systems::Simulator<double> simulator(*diagram);

  //lcm.StartReceiveThread();
  simulator.set_target_realtime_rate(1);
  //simulator.Initialize();
  //simulator.set_publish_every_time_step(true);
  simulator.StepTo(20);
}

}  // namespace systems
}  // namespace drake

int main() {
  drake::systems::DoMain();
  return 0;
}