/**
 * @file initialization.h
 * @author Miguel Sánchez Tello
 * @date 26/06/2016
 * @brief Header file for the initialization of the individuals
 *
 */

#ifndef INITIALIZATION_H
#define INITIALIZATION_H

/********************************* Includes *******************************/

#include "individual.h" // Individual

/********************************* Methods ********************************/

/**
 * @brief Allocates memory for parents and children. Also, they are initialized
 * @param totalIndividuals The total number of individuals (parents + children)
 * @param nObjectives The number of objectives
 * @param nFeatures The number of features (columns) of the database
 * @param maxFeatures The maximum number of features that will be set to "1"
 * @return Pointer to the population
 */
individual* initPopulation(const int totalIndividuals, const unsigned char nObjectives, const int nFeatures, const int maxFeatures);

#endif