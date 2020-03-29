#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
    num_particles = 100;  // TODO: Verify this parameter, is processing taking too long?

    // Initialise with a gaussian based on GPS
    std::default_random_engine gen;

    double std_x = std[0];
    double std_y = std[1];
    double std_theta = std[2];

    // Define gaussian values
    std::normal_distribution<double> dist_x(x, std_x);
    std::normal_distribution<double> dist_y(y, std_y);
    std::normal_distribution<double> dist_theta(theta, std_theta);

    for (int i = 0; i < num_particles; i++)
    {
        // Initialise instance of particle.
        Particle particle;

        // Initialise particles with Gaussian drawn from GPS initialisation.
        particle.id = i;
        particle.x = dist_x(gen);
        particle.y = dist_y(gen);
        particle.theta = dist_theta(gen);

        // Initialise weight to standard value
        particle.weight = init_weight;

        particles.push_back(particle);
        weights.push_back(1.0);
    }

    // Mark as initialised
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
    // Define a gaussian distribution of noise with a mea[ of zero and a pre defined StdDev
    std::default_random_engine gen;

    int mean = 0;
    double std_x = std_pos[0];
    double std_y = std_pos[1];
    double std_theta = std_pos[2];

    std::normal_distribution<double> dist_x(mean, std_x);
    std::normal_distribution<double> dist_y(mean, std_y);
    std::normal_distribution<double> dist_theta(mean, std_theta);

    // Update all particles with predictions
    for (int i = 0; i < num_particles; i++)
    {
        // Define Predictions
        double x_pred;
        double y_pred;
        double theta_pred;

        // Define particle components
        double x_0 = particles[i].x;
        double y_0 = particles[i].y;
        double theta = particles[i].theta;

        // Update predictions based on yaw rate
        if (fabs(yaw_rate) > 0.0001)
        {
            x_pred = x_0 + (velocity / yaw_rate) * (std::sin((theta + (yaw_rate * delta_t)) - std::sin(theta)));
            y_pred = y_0 + (velocity / yaw_rate) * (std::cos(theta) - std::cos((theta + (yaw_rate * delta_t))));
            theta_pred = theta + (yaw_rate * delta_t);
        }
        else
        {
            x_pred = x_0 + (velocity * delta_t) * std::cos(theta);
            y_pred = y_0 + (velocity * delta_t) + std::sin(theta);
            theta_pred = theta;
        }

        // Update particles including gaussian noise
        particles[i].x = x_pred + dist_x(gen);
        particles[i].y = y_pred + dist_y(gen);
        particles[i].theta = theta_pred + dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, vector<LandmarkObs>& observations)
{
    for (unsigned int i = 0; i < observations.size(); i++)
    {
        LandmarkObs obs = observations[i];

        // Initialise varibale to store new ID.
        int new_id = obs.id;

        // Landmark struct defines a double, therefore initalise with max value of double.
        double current_euclidian_distance = std::numeric_limits<double>::max();

        // Iterate over each prediction to find nearest neighbour
        for (unsigned int j = 0; j < predicted.size(); j++)
        {
            LandmarkObs pred = predicted[j];

            double euclidian_distance = dist(obs.x, obs.y, pred.x, pred.y);

            if (euclidian_distance < current_euclidian_distance)
            {
                new_id = pred.id;
                current_euclidian_distance = euclidian_distance;
            }
        }

        // Update observation
        observations[i].id = new_id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], const vector<LandmarkObs>& observations,
                                   const Map& map_landmarks)
{
    for (unsigned int i = 0; i < num_particles; i++)
    {
        // Assign individual components to matrix transform.
        double xp = particles[i].x;
        double yp = particles[i].y;
        double theta = particles[i].theta;

        vector<LandmarkObs> tf_observations;

        for (int j = 0; j < observations.size(); j++)
        {
            // Transform observation to map coords, xp = particles, xc = observations
            LandmarkObs tf_observation;

            double xc = observations[j].x;
            double yc = observations[j].y;

            double xm = xp + (std::cos(theta) * xc) - (std::sin(theta) * yc);
            double ym = yp + (std::sin(theta) * xc) + (std::cos(theta) * yc);

            tf_observation.x = xm;
            tf_observation.y = ym;
            tf_observation.id = j; //TODO: THIS IS ZERO FOR SOME REASON!

            tf_observations.push_back(tf_observation);
        }

        // Filter landmarks to only that which is within the sensor range
        vector<LandmarkObs> landmarks_in_range;

        for (int k = 0; k < map_landmarks.landmark_list.size(); k++)
        {
            // Calculate Euclidian Distance to Landmark
            double distance = dist(xp, yp, map_landmarks.landmark_list[k].x_f, map_landmarks.landmark_list[k].y_f);

            // Filter to only that distance this is within the sensor range
            if (distance <= sensor_range)
            {
                LandmarkObs landmark_in_range;
                landmark_in_range.x = map_landmarks.landmark_list[k].x_f;
                landmark_in_range.y = map_landmarks.landmark_list[k].y_f;
                landmark_in_range.id = map_landmarks.landmark_list[k].id_i;

                landmarks_in_range.push_back(landmark_in_range);
            }
        }

        // Use dataAssociation to update the observations with the ID of the nearest landmark
        dataAssociation(landmarks_in_range, tf_observations);

        // Calculate the weight of each particle using multivariate gaussian probability density function
        double sig_x = std_landmark[0];
        double sig_y = std_landmark[1];
        double var_x = sig_x * sig_x;
        double var_y = sig_y * sig_y;
        double gaussian_norm = 1.0 / (2.0 * M_PI * sig_x * sig_y);

        // Reset the particles weight
        particles[i].weight = 1.0;

        double probability = 1.0;

        for (unsigned int j = 0; j < tf_observations.size(); j++)
        {
            double tf_x = tf_observations[j].x;
            double tf_y = tf_observations[j].y;
            double tf_id = tf_observations[j].id;

            for (unsigned int k = 0; k < landmarks_in_range.size(); k++)
            {
                double ldmk_x = landmarks_in_range[k].x;
                double ldmk_y = landmarks_in_range[k].y;
                double ldmk_id = landmarks_in_range[k].id;

                // Find matching landmarks and observations
                if (tf_id == ldmk_id)
                {
                    double mu_x = ldmk_x;
                    double mu_y = ldmk_y;

                    // Calculate the gaussian probability for each observation and obtain the product
                    double exponent_x = ((tf_x - mu_x) * (tf_x - mu_x)) / (2.0 * var_x);
                    double exponent_y = ((tf_y - mu_y) * (tf_y - mu_y)) / (2.0 * var_y);

                    probability = probability * gaussian_norm * exp(-(exponent_x + exponent_y));
                    std::cout << (exponent_x + exponent_y) << std::endl;
                    break;

                }
            }
        }

        particles[i].weight = probability;
        weights[i] = probability;
    }

    double weight_normalizer = std::accumulate(weights.begin(), weights.end(),0.0f);

    for (int p = 0; p < num_particles; ++p){
        particles[p].weight = particles[p].weight / weight_normalizer;
        weights[p] = particles[p].weight;
    }
}

void ParticleFilter::resample()
{
    // Sampled particle set
    vector<Particle> sampled_particles(num_particles);

    std::default_random_engine gen;
    std::discrete_distribution <int> sample(weights.begin(), weights.end());

    for ( int i = 0; i < num_particles; ++i){
        int k = sample(gen);
        sampled_particles.push_back(particles[k]);
    }

    particles = sampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, const vector<int>& associations, const vector<double>& sense_x,
                                     const vector<double>& sense_y)
{
    // particle: the particle to which assign each listed association,
    //   and association's (x,y) world coordinates mapping
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord)
{
    vector<double> v;

    if (coord == "X")
    {
        v = best.sense_x;
    }
    else
    {
        v = best.sense_y;
    }

    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}