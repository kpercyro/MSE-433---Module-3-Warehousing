# MSE 433 - Module 3: Warehouse Conveyor Belt Optimization

Case study for MSE 433 exploring order consolidation on a 4-belt conveyor loop system at the IDEAS Clinic.

## Problem

A warehouse conveyor system has 4 belts forming a loop. Items are loaded onto a ramp, circulate on the conveyor, and pneumatic arms push items off when a scanner detects a match with the active order on that belt. The goal is to minimize the total time (makespan) to fulfill all orders by optimizing:

1. **Belt assignment** - which orders go on which belt and in what sequence
2. **Tote loading order** - the sequence items are physically placed on the ramp

## Repository Structure

```
MSE433_M3_data_generator.ipynb   # Provided data generation notebook (seed=100)
MSE433_M3_Example input.csv      # Example conveyor input format
MSE433_M3_Example output.csv     # Example conveyor output format
jeevan/                          # Jeevan's optimization solution
kate/                            # Kate's optimization solution
```

## Team Members

- Jeevan Parmar
- Kate
