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

static default_random_engine gen;

template<typename Collection, typename unop>
vector<double> map2(Collection col, unop op) {
  vector<double> result(col.size());
  transform(col.begin(), col.end(), result.begin(), op);
  return result;
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  num_particles = 50;

  for (int i = 0; i < num_particles; i++) {
    Particle particle;
    particle.id = i;
    particle.x = x;
    particle.y = y;
    particle.theta = theta;
    particle.weight = 1;
    addNoise(particle, std);
    particles.push_back(particle);
  }

  is_initialized = true;
}

void ParticleFilter::addNoise(Particle &particle, double std[]) {
  normal_distribution<double> randn(0.0, 1.0);
  particle.x += randn(gen) * std[0];
  particle.y += randn(gen) * std[1];
  particle.theta += randn(gen) * std[2];
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  for (Particle &particle : particles) {
    predictParticle(particle, delta_t, std_pos, velocity, yaw_rate);
  }
}

void ParticleFilter::predictParticle(Particle &particle, double delta_t,
                                     double std_pos[], double velocity,
                                     double yaw_rate) {

  if (fabs(yaw_rate) < 0.0001) {
    particle.x += velocity * delta_t * cos(particle.x);
    particle.y += velocity * delta_t * sin(particle.x);
  } else {
    double new_theta = particle.theta + yaw_rate * delta_t;
    particle.x += velocity / yaw_rate * (sin(new_theta) - sin(particle.theta));
    particle.y += velocity / yaw_rate * (cos(particle.theta) - cos(new_theta));
    particle.theta = new_theta;
  }
  addNoise(particle, std_pos);
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {

  for (Particle &particle : particles) {
    particle.weight = getWeight(particle, std_landmark, observations,
                                map_landmarks);
  }
}

double ParticleFilter::getWeight(const Particle &particle,
                                 double std_landmark[],
                                 const vector<LandmarkObs> &observations,
                                 const Map &map_landmarks) {

  vector<double> weightsForObservations = getWeightsForObservations(
      particle, std_landmark, observations, map_landmarks);
  return multiply(weightsForObservations);
}

vector<double> ParticleFilter::getWeightsForObservations(
    const Particle &particle, double std_landmark[],
    const vector<LandmarkObs> &observations, const Map &map_landmarks) {

  auto getWeightForObservation =
      [&]
      (const LandmarkObs &observation) {
        LandmarkObs obsInMapCoords = getObsInMapCoords(particle, observation);
        LandmarkObs best_landmark = getLandmarkBestMatchingObs(obsInMapCoords, map_landmarks);
        return getWeight(obsInMapCoords, best_landmark, std_landmark);
      };

  return map2(observations, getWeightForObservation);
}

double ParticleFilter::multiply(const vector<double> &numbers) {
  return accumulate(begin(numbers), end(numbers), 1.0, multiplies<double>());
}

vector<double> ParticleFilter::getWeightsOfParticles() {
  return map2(particles, [](Particle particle) {return particle.weight;});
}

void ParticleFilter::resample() {
  vector<double> weights = getWeightsOfParticles();
  discrete_distribution<int> weight_distribution(weights.begin(),
                                                 weights.end());
  vector<Particle> new_particles;

  for (int i = 0; i < num_particles; i++) {
    new_particles.push_back(particles[weight_distribution(gen)]);
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
      *min_element(
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
