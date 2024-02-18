from source.fit_model import FitModel
import sys


# defines the model as a function of the two parameters and the independent variable, here Mayo-Lewis terminal model
def model(parameters, f2):
    r1 = parameters[1]
    r2 = parameters[0]
    return (r2*f2**2.0 + f2*(1.0-f2))/(r2*f2**2.0 + 2*f2*(1.0-f2) + r1*(1-f2)**2.0)

# defines the derivative of the model with respect to the data
def dmodel(parameters, f2):
    r1 = parameters[1]
    r2 = parameters[0]
    return (2.0*r2*f2-2*f2 + 1.0)/(r2*f2**2.0 + 2*f2*(1.0-f2) + r1*(1-f2)**2.0) - (r2*f2**2.0+f2*(1.0-f2))*(r2*f2**2.0 + 2*f2*(1.0-f2) + r1*(1-f2)**2.0)**(-2.0)*(2*f2*r2+2-4.0*f2-2.0*r1*(1.0-f2))


if len(sys.argv) > 1:
    config = str(sys.argv[1])
    fit = FitModel(model, dmodel, config)
else:
    fit = FitModel(model, dmodel, "input.ini")

