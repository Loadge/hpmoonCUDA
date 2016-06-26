/**
 * @file tournament.h
 * @author Miguel Sánchez Tello
 * @date 26/06/2016
 * @brief Header file that defines functions for the realization of the tournament (selection of individuals)
 *
 */

#ifndef TOURNAMENT_H
#define TOURNAMENT_H

/********************************* Methods ********************************/

/**
 * @brief Competition between randomly selected individuals. The best individuals are stored in the pool
 * @param pool Position of the selected individuals for the crossover
 * @param poolSize Number of selected individuals for the crossover
 * @param tourSize Number of individuals competing in the tournament
 * @param populationSize The size of the population
 */
void fillPool(int *pool, const int poolSize, const int tourSize, const int populationSize);

#endif