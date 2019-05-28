import numpy as np

def SGD(del_errors,weighs,a=0.1):
	delta_error2,delta_error1 = del_errors[1],del_errors[0]
	weights1,weights0 = weighs[1],weighs[0]
	weights1 -= a*delta_error2
	weights0 -= a*delta_error1
	return weights1,weights0

def momentum(del_errors,weighs,velocity,a=0.1,B=0.9):
	delta_error2,delta_error1 = del_errors[1],del_errors[0]
	weights1,weights0 = weighs[1],weighs[0]
	velocity[1] = B*velocity[1] + (1 - B)*delta_error2
	velocity[0] = B*velocity[0] + (1 - B)*delta_error1
	weights1 -= a*velocity[1]
	weights0 -= a*velocity[0]
	return weights1,weights0,velocity
def adagrad(del_errors,weighs,S,a=0.01,E=10**(-7)):
	delta_error2,delta_error1 = del_errors[1],del_errors[0]
	weights1,weights0 = weighs[1],weighs[0]
	S[1] += np.square(delta_error2)
	S[0] += np.square(delta_error1)
	weights1 -= (a/(np.sqrt(S[1]+E)))*delta_error2
	weights0 -= (a/(np.sqrt(S[0]+E)))*delta_error1
	return weights1,weights0,S

def rmsprop(del_errors,weighs,S,B=0.9,a=0.001,E=10**(-6)):
	delta_error2,delta_error1 = del_errors[1],del_errors[0]
	weights1,weights0 = weighs[1],weighs[0]
	S[1] = B*S[1] +  (1-B)*np.square(delta_error2)
	S[0] = B*S[0] +  (1-B)*np.square(delta_error1)
	weights1 -= (a/(np.sqrt(S[1]+E)))*delta_error2
	weights0 -= (a/(np.sqrt(S[0]+E)))*delta_error1
	return weights1,weights0,S

def adadelta(del_errors,weighs,S,D,B=0.9,a=0.001,E=10**(-6)):
	delta_error2,delta_error1 = del_errors[1],del_errors[0]
	weights1,weights0 = weighs[1],weighs[0]
	S[1] = B*S[1] +  (1-B)*np.square(delta_error2)
	S[0] = B*S[0] +  (1-B)*np.square(delta_error1)
	weights1 -= ((np.sqrt(D[1] + E))/(np.sqrt(S[1]+E)))*delta_error2
	weights0 -= ((np.sqrt(D[0] + E))/(np.sqrt(S[0]+E)))*delta_error1
	D[1] = B*D[1] + (1 - B)*(np.square(weights1 - weighs[1]))
	D[0] = B*D[0] + (1 - B)*(np.square(weights0 - weighs[0]))
	return weights1,weights0,S,D

def adam(del_errors,weighs,velocity,S,a=0.01,E=10**(-7),beta=[0.9,0.999]):
	delta_error2,delta_error1 = del_errors[1],del_errors[0]
	weights1,weights0 = weighs[1],weighs[0]
	S[1] = beta[1]*S[1]	+ (1 - beta[1])*(np.square(delta_error2))
	S[0] = beta[1]*S[0]	+ (1 - beta[1])*(np.square(delta_error1))
	velocity[1] = beta[0]*velocity[1] + ((1 - beta[0])*(delta_error2))
	velocity[0] = beta[0]*velocity[0] + ((1 - beta[0])*(delta_error1))
	s_cap1 = S[1]/(1 - beta[1])
	s_cap0 = S[0]/(1 - beta[1])
	v_cap1 = velocity[1]/(1 - beta[1])
	v_cap0 = velocity[0]/(1 - beta[1])
	weights1 -= (a/(np.sqrt(s_cap1) + E ))*v_cap1
	weights0 -= (a/(np.sqrt(s_cap0) + E ))*v_cap0
	return weights1,weights0,S,velocity
def adamax(del_errors,weighs,velocity,S,a=0.002,E=10**(-7),beta=[0.9,0.999]):
	delta_error2,delta_error1 = del_errors[1],del_errors[0]
	weights1,weights0 = weighs[1],weighs[0]
	S[1] = max(np.mean(beta[1]*S[1])	, np.abs(np.mean(delta_error2)))
	S[0] = max(np.mean(beta[1]*S[0])	, np.abs(np.mean(delta_error1)))
	velocity[1] = beta[0]*velocity[1] + ((1 - beta[0])*(delta_error2))
	velocity[0] = beta[0]*velocity[0] + ((1 - beta[0])*(delta_error1))
	v_cap1 = velocity[1]/(1 - beta[1])
	v_cap0 = velocity[0]/(1 - beta[1])
	weights1 -= (a/((S[1]) ))*v_cap1
	weights0 -= (a/((S[0])))*v_cap0
	return weights1,weights0,S,velocity
def amsgrad(del_errors,weighs,velocity,S,a=0.001,E=10**(-7),beta=[0.9,0.999]):
	delta_error2,delta_error1 = del_errors[1],del_errors[0]
	weights1,weights0 = weighs[1],weighs[0]
	S[1] = beta[1]*S[1]	+ (1 - beta[1])*(np.square(delta_error2))
	S[0] = beta[1]*S[0]	+ (1 - beta[1])*(np.square(delta_error1))
	velocity[1] = beta[0]*velocity[1] + ((1 - beta[0])*(delta_error2))
	velocity[0] = beta[0]*velocity[0] + ((1 - beta[0])*(delta_error1))
	v_cap1 = velocity[1]/(1 - beta[1])
	v_cap0 = velocity[0]/(1 - beta[1])
	weights1 -= (a/(np.sqrt(S[1]) + E))*v_cap1
	weights0 -= (a/(np.sqrt(S[0]) + E ))*v_cap0
	return weights1,weights0,S,velocity