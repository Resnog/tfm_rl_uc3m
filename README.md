# Master's Final Proyect in Robotics at the UC3M

This repo contains the code used to train an RL agent that controls the actuator/tooltip point of a robot arm in a mobile manipulator to pour liquids and masses of small solids (like rice), using a research assistencial robot of the UC3M named HEROITEA.

The agent was trained on a simulation environment built in Coppelia Sim, with the usage of the PyRep library. This one is better suited for RL agent's experimentation with CoppeliaSim.

The current version of the project, bounded by PyRep requires an Ubuntu distro to be used by the user. Definitely won't recomend it's use in Fedora/CentOS/RHEL since installing the library will break OpenSSL dependencies, you will be having tons of fun guessing what happened and reinstalling everything until you decide to reformat your system at the end.

### Problem's overview

The main goal of the project was to learn how to use just one joint to pour a glass of fluids/food into other identical glass. The joint's interval was defined empirically, but it's dependant of the angle and the nature of the fluid, such as density and viscosity (or if it's just a mass of little particles). 

The metric used in the case was the number of particles that reached the goal destination successfully, this is just a proof of concept and in no way will make wonrders when puring anything into a glass. But the proof of concept was successful and could now advance to more meticulous phases of research and development.

### Algoritms implemented

The RL agent was designed with multiple agent testing, you can download the code, add a new agent class and begin having fun.

#### More to come

When I have my Ubuntu machine with me, I'll organize properly the code since it can be a little tedious to work with initially.
