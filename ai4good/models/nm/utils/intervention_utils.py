from collections import defaultdict


class Interventions:
    def __init__(self):
        self.checkpoints = defaultdict(list)

    def add(self, g_intervention, time, **kwargs):
        """ Adding an intervention to the checkpoints dictionary requires
            a graph and a time step at which the intervention will occur.
            Other parameters that are altered are optional """

        # We assume that interventions are added in order
        self.checkpoints["G"].append(g_intervention)
        self.checkpoints["t"].append(time)

        # Update the other parameters
        for parameter, value in kwargs.items():
            self.checkpoints[parameter].append(value)

        # If there are parameters that haven't been updated in this time step,
        # keep their previous values
        for parameter, value in self.checkpoints.items():
            if len(value) != len(self.checkpoints["t"]):
                value.append(value[-1])

    def edit(self, g_intervention, time, **kwargs):
        """ Edit an intervention at a given time step in the checkpoints dictionary """
        # Raise an exception if the time is not in the checkpoints
        if time not in self.checkpoints["t"]:
            raise Exception("Time stamp not found in interventions!")
        else:
            # Get the index of the intervention in the checkpoint
            t_index = self.checkpoints["t"].index(time)

            # Replace new network intervention
            self.checkpoints["G"][t_index] = g_intervention

            # Edit all parameters at that index
            for parameter, value in kwargs.items():
                self.checkpoints[parameter][t_index] = value

    def remove(self, time):
        """ Remove an intervention at a given time step in the checkpoints dictionary"""
        # Raise an exception if the time is not in the checkpoints
        if time not in self.checkpoints["t"]:
            raise Exception("Time stamp not found in interventions!")
        else:
            # Get the index of the intervention in the checkpoint
            t_index = self.checkpoints["t"].index(time)

            # Remove the intervention done at a specific time
            for parameter, value in self.checkpoints.items():
                value.pop(t_index)

    def clear(self):
        self.checkpoints.clear()

    def get_checkpoints(self):
        return dict(self.checkpoints)


