import pandas as pd
from hosvd import hosvd
import torch
from sktensor.tucker import hooi
from sktensor import dtensor


class Recurse(Exception):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

def recurse(*args, **kwargs):
    raise Recurse(*args, **kwargs)
        
def tail_recursive(f):
    def decorated(*args, **kwargs):
        while True:
            try:
                return f(*args, **kwargs)
            except Recurse as r:
                args = r.args
                kwargs = r.kwargs
                continue
    return decorated

@tail_recursive
def data_function (f, a, c, v, t, scale=1):
	freq_centroid = start_frequency + (f * scale * increments_frequency)
	freq_start = freq_centroid - (f * scale * increments_frequency)
	freq_end = freq_centroid + (f * scale * increments_frequency)
	freq_sel = (data['frequency'] >= freq_start) & (data['frequency'] < freq_end)

	ang_centroid = start_angle + (a * scale * increments_angle)
	ang_start = ang_centroid - (a * scale * increments_angle)
	ang_end = ang_centroid + (a * scale * increments_angle)
	ang_sel = (data['angle_of_attack'] >= ang_start) & (data['angle_of_attack'] < ang_end)

	clen_centroid = start_chordlen + (c * scale * increments_chordlen)
	clen_start = clen_centroid - (c * scale * increments_chordlen)
	clen_end = clen_centroid + (c * scale * increments_chordlen)
	clen_sel = (data['chord_len'] >= clen_start) & (data['chord_len'] < clen_end)

	vel_centroid = start_fsvelocity + (v * scale * increments_fsvelocity)
	vel_start = vel_centroid - (v * scale * increments_fsvelocity)
	vel_end = vel_centroid + (v * scale * increments_fsvelocity)
	vel_sel = (data['free_stream_velocity'] >= vel_start) & (data['free_stream_velocity'] < vel_end)

	thick_centroid = start_thickness + (t * scale * increments_thickness)
	thick_start = thick_centroid - (t * scale * increments_thickness)
	thick_end = thick_centroid + (t * scale * increments_thickness)
	thick_sel = (data['thickness'] >= thick_start) & (data['thickness'] < thick_end)

	retval = data[freq_sel & ang_sel & clen_sel & vel_sel & thick_sel]
	if (retval.empty):
		recurse(f,a,c,v,t,scale+1)
	else:
		return (retval, retval['sounddb'].median())

print('---')

data = pd.read_csv('airfoil.csv')

gradations = [10,10,10,20,10]

start_frequency = data['frequency'].min()
end_frequency = data['frequency'].max()
increments_frequency = (end_frequency - start_frequency) / gradations[0]

start_angle = data['angle_of_attack'].min()
end_angle = data['angle_of_attack'].max()
increments_angle = (end_angle - start_angle) / gradations[1]

start_chordlen = data['chord_len'].min()
end_chordlen = data['chord_len'].max()
increments_chordlen = (end_chordlen - start_chordlen) / gradations[2]

start_fsvelocity = data['free_stream_velocity'].min()
end_fsvelocity = data['free_stream_velocity'].max()
increments_fsvelocity = (end_fsvelocity - start_fsvelocity) / gradations[3]

start_thickness = data['thickness'].min()
end_thickness = data['thickness'].max()
increments_thickness = (end_thickness - start_thickness) / gradations[4]


# setup toolbar
import sys
toolbar_width = 400
sys.stdout.write("[%s]" % (" " * toolbar_width))
sys.stdout.flush()
sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

A = torch.zeros(gradations[0], gradations[1], gradations[2], gradations[3], gradations[4], dtype=torch.float64)

for f in range(gradations[0]):
	for a in range(gradations[1]):
		for c in range(gradations[2]):
			for v in range(gradations[3]):
				for t in range(gradations[4]):
					A[f,a,c,v,t] = data_function(f+1, a+1, c+1, v+1, t+1,1)[1]
		sys.stdout.write("-")
		sys.stdout.flush()
sys.stdout.write("]\n") # this ends the progress bar

torch.save(A, 'A3.pt')

## x = torch.tensor([[1,2], [3,4]], dtype=torch.float64)
## print(x)
## S, Ulist = hosvd(x)
## print(S)
## print(Ulist[0])
## print(Ulist[1])
## print(len(Ulist))

## y = hooi(dtensor(x), [1,2], init='nvecs')
## print(y)



