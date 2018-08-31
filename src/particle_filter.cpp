/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  num_particles = 50;
  default_random_engine gen;
  normal_distribution<double> randn(0.0, 1.0);

  for (int i = 0; i < num_particles; i++) {
    Particle particle;
    particle.id = i;
    particle.x = x + randn(gen) * std[0];
    particle.y = y + randn(gen) * std[1];
    particle.theta = theta + randn(gen) * std[2];
    particle.weight = 1;
    particles.push_back(particle);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  default_random_engine gen;

  double new_theta;
  double new_x_mean;
  double new_y_mean;

  for (int i = 0; i < num_particles; i++) {
    if (fabs(yaw_rate) < 0.0001) {
      new_theta = particles[i].theta;
      new_x_mean = particles[i].x + (velocity * delta_t * cos(new_theta));
      new_y_mean = particles[i].y + (velocity * delta_t * sin(new_theta));
    } else {
      new_theta = particles[i].theta + yaw_rate * delta_t;
      new_x_mean = particles[i].x
          + (velocity / yaw_rate) * (sin(new_theta) - sin(particles[i].theta));
      new_y_mean = particles[i].y
          + (velocity / yaw_rate) * (cos(particles[i].theta) - cos(new_theta));
    }

    normal_distribution<double> randn(0.0, 1.0);

    particles[i].x = new_x_mean + randn(gen) * std_pos[0];
    particles[i].y = new_y_mean + randn(gen) * std_pos[1];
    particles[i].theta = new_theta + randn(gen) * std_pos[2];
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {

  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html

  LandmarkObs obsInMapCoords;

  for (int i = 0; i < particles.size(); i++) {
    double prob = 1.;

    for (int k = 0; k < observations.size(); k++) {
      obsInMapCoords = getObsInMapCoords(particles[i], observations[k]);
      prob *= getWeight(
          obsInMapCoords,
          getLandmarkBestMatchingObs(obsInMapCoords, map_landmarks),
          std_landmark);
    }
    particles[i].weight = prob;
  }
}

std::vector<double> ParticleFilter::getWeightsOfParticles() {
  std::vector<double> weights(particles.size());
  std::transform(particles.begin(), particles.end(), weights.begin(),
                 [](Particle particle) {return particle.weight;});
  return weights;
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  std::vector<Particle> new_particles;
  int index;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::vector<double> weights = getWeightsOfParticles();
  std::discrete_distribution<int> weight_distribution(weights.begin(),
                                                      weights.end());

  for (int i = 0; i < num_particles; i++) {
    index = weight_distribution(gen);
    new_particles.push_back(particles[index]);
  }
  particles = new_particles;
}

LandmarkObs ParticleFilter::getObsInMapCoords(const Particle &part,
                                              const LandmarkObs &obs) {
  double x = part.x + obs.x * cos(part.theta) - obs.y * sin(part.theta);
  double y = part.y + obs.x * sin(part.theta) + obs.y * cos(part.theta);
  return LandmarkObs { x, y };
}

LandmarkObs ParticleFilter::getLandmarkBestMatchingObs(
    const LandmarkObs &obs, const Map &map_landmarks) {

  Map::single_landmark_s min =
      *std::min_element(
          map_landmarks.landmark_list.begin(),
          map_landmarks.landmark_list.end(),
          [obs](const Map::single_landmark_s &landmark1, const Map::single_landmark_s &landmark2)
          {
            double distance1 = dist(obs.x, obs.y, landmark1.x_f, landmark1.y_f);
            double distance2 = dist(obs.x, obs.y, landmark2.x_f, landmark2.y_f);
            return distance1 < distance2;
          });

  return LandmarkObs { min.x_f, min.y_f };
}

double ParticleFilter::gauss(double x, double mean, double stddev) {
  double dx = x - mean;
  double var = stddev * stddev;
  return 1.0 / (stddev * sqrt(2.0 * M_PI)) * exp(-dx * dx / (2.0 * var));
}

double ParticleFilter::getWeight(const LandmarkObs &obs,
                                 const LandmarkObs &best_landmark,
                                 double std_landmark[]) {

  return gauss(obs.x, best_landmark.x, std_landmark[0])
      * gauss(obs.y, best_landmark.y, std_landmark[1]);
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
