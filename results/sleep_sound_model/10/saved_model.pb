·Á
&Þ%
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
¼
AvgPool

value"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


DepthwiseConv2dNative

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

,
Exp
x"T
y"T"
Ttype:

2
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
­
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
S
Imag

input"T
output"Tout"
Ttype0:
2"
Touttype0:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	
0
Neg
x"T
y"T"
Ttype:
2
	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
8
Pow
x"T
y"T
z"T"
Ttype:
2
	
e
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2		
@
ReadVariableOp
resource
value"dtype"
dtypetype
S
Real

input"T
output"Tout"
Ttype0:
2"
Touttype0:
2
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
-
Sqrt
x"T
y"T"
Ttype:

2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨

StatelessWhile

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint

@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
«
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type*
output_handleéelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements#
handleéelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.7.02v2.7.0-rc1-69-gc256c071bb28Ë

audio_classifier/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameaudio_classifier/dense/kernel

1audio_classifier/dense/kernel/Read/ReadVariableOpReadVariableOpaudio_classifier/dense/kernel*
_output_shapes

:*
dtype0

audio_classifier/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameaudio_classifier/dense/bias

/audio_classifier/dense/bias/Read/ReadVariableOpReadVariableOpaudio_classifier/dense/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h
kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*
shared_namekernel
a
kernel/Read/ReadVariableOpReadVariableOpkernel*
_output_shapes

:(*
dtype0
À
.audio_classifier/leaf/learnable_pooling/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*?
shared_name0.audio_classifier/leaf/learnable_pooling/kernel
¹
Baudio_classifier/leaf/learnable_pooling/kernel/Read/ReadVariableOpReadVariableOp.audio_classifier/leaf/learnable_pooling/kernel*&
_output_shapes
:(*
dtype0

 audio_classifier/leaf/PCEN/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*1
shared_name" audio_classifier/leaf/PCEN/alpha

4audio_classifier/leaf/PCEN/alpha/Read/ReadVariableOpReadVariableOp audio_classifier/leaf/PCEN/alpha*
_output_shapes
:(*
dtype0

 audio_classifier/leaf/PCEN/deltaVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*1
shared_name" audio_classifier/leaf/PCEN/delta

4audio_classifier/leaf/PCEN/delta/Read/ReadVariableOpReadVariableOp audio_classifier/leaf/PCEN/delta*
_output_shapes
:(*
dtype0

audio_classifier/leaf/PCEN/rootVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*0
shared_name!audio_classifier/leaf/PCEN/root

3audio_classifier/leaf/PCEN/root/Read/ReadVariableOpReadVariableOpaudio_classifier/leaf/PCEN/root*
_output_shapes
:(*
dtype0
¢
%audio_classifier/leaf/PCEN/EMA/smoothVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*6
shared_name'%audio_classifier/leaf/PCEN/EMA/smooth

9audio_classifier/leaf/PCEN/EMA/smooth/Read/ReadVariableOpReadVariableOp%audio_classifier/leaf/PCEN/EMA/smooth*
_output_shapes
:(*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
¤
$Adam/audio_classifier/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*5
shared_name&$Adam/audio_classifier/dense/kernel/m

8Adam/audio_classifier/dense/kernel/m/Read/ReadVariableOpReadVariableOp$Adam/audio_classifier/dense/kernel/m*
_output_shapes

:*
dtype0

"Adam/audio_classifier/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/audio_classifier/dense/bias/m

6Adam/audio_classifier/dense/bias/m/Read/ReadVariableOpReadVariableOp"Adam/audio_classifier/dense/bias/m*
_output_shapes
:*
dtype0
v
Adam/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*
shared_nameAdam/kernel/m
o
!Adam/kernel/m/Read/ReadVariableOpReadVariableOpAdam/kernel/m*
_output_shapes

:(*
dtype0
Î
5Adam/audio_classifier/leaf/learnable_pooling/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*F
shared_name75Adam/audio_classifier/leaf/learnable_pooling/kernel/m
Ç
IAdam/audio_classifier/leaf/learnable_pooling/kernel/m/Read/ReadVariableOpReadVariableOp5Adam/audio_classifier/leaf/learnable_pooling/kernel/m*&
_output_shapes
:(*
dtype0
¦
'Adam/audio_classifier/leaf/PCEN/alpha/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*8
shared_name)'Adam/audio_classifier/leaf/PCEN/alpha/m

;Adam/audio_classifier/leaf/PCEN/alpha/m/Read/ReadVariableOpReadVariableOp'Adam/audio_classifier/leaf/PCEN/alpha/m*
_output_shapes
:(*
dtype0
¦
'Adam/audio_classifier/leaf/PCEN/delta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*8
shared_name)'Adam/audio_classifier/leaf/PCEN/delta/m

;Adam/audio_classifier/leaf/PCEN/delta/m/Read/ReadVariableOpReadVariableOp'Adam/audio_classifier/leaf/PCEN/delta/m*
_output_shapes
:(*
dtype0
¤
&Adam/audio_classifier/leaf/PCEN/root/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*7
shared_name(&Adam/audio_classifier/leaf/PCEN/root/m

:Adam/audio_classifier/leaf/PCEN/root/m/Read/ReadVariableOpReadVariableOp&Adam/audio_classifier/leaf/PCEN/root/m*
_output_shapes
:(*
dtype0
°
,Adam/audio_classifier/leaf/PCEN/EMA/smooth/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*=
shared_name.,Adam/audio_classifier/leaf/PCEN/EMA/smooth/m
©
@Adam/audio_classifier/leaf/PCEN/EMA/smooth/m/Read/ReadVariableOpReadVariableOp,Adam/audio_classifier/leaf/PCEN/EMA/smooth/m*
_output_shapes
:(*
dtype0
¤
$Adam/audio_classifier/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*5
shared_name&$Adam/audio_classifier/dense/kernel/v

8Adam/audio_classifier/dense/kernel/v/Read/ReadVariableOpReadVariableOp$Adam/audio_classifier/dense/kernel/v*
_output_shapes

:*
dtype0

"Adam/audio_classifier/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/audio_classifier/dense/bias/v

6Adam/audio_classifier/dense/bias/v/Read/ReadVariableOpReadVariableOp"Adam/audio_classifier/dense/bias/v*
_output_shapes
:*
dtype0
v
Adam/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*
shared_nameAdam/kernel/v
o
!Adam/kernel/v/Read/ReadVariableOpReadVariableOpAdam/kernel/v*
_output_shapes

:(*
dtype0
Î
5Adam/audio_classifier/leaf/learnable_pooling/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*F
shared_name75Adam/audio_classifier/leaf/learnable_pooling/kernel/v
Ç
IAdam/audio_classifier/leaf/learnable_pooling/kernel/v/Read/ReadVariableOpReadVariableOp5Adam/audio_classifier/leaf/learnable_pooling/kernel/v*&
_output_shapes
:(*
dtype0
¦
'Adam/audio_classifier/leaf/PCEN/alpha/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*8
shared_name)'Adam/audio_classifier/leaf/PCEN/alpha/v

;Adam/audio_classifier/leaf/PCEN/alpha/v/Read/ReadVariableOpReadVariableOp'Adam/audio_classifier/leaf/PCEN/alpha/v*
_output_shapes
:(*
dtype0
¦
'Adam/audio_classifier/leaf/PCEN/delta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*8
shared_name)'Adam/audio_classifier/leaf/PCEN/delta/v

;Adam/audio_classifier/leaf/PCEN/delta/v/Read/ReadVariableOpReadVariableOp'Adam/audio_classifier/leaf/PCEN/delta/v*
_output_shapes
:(*
dtype0
¤
&Adam/audio_classifier/leaf/PCEN/root/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*7
shared_name(&Adam/audio_classifier/leaf/PCEN/root/v

:Adam/audio_classifier/leaf/PCEN/root/v/Read/ReadVariableOpReadVariableOp&Adam/audio_classifier/leaf/PCEN/root/v*
_output_shapes
:(*
dtype0
°
,Adam/audio_classifier/leaf/PCEN/EMA/smooth/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*=
shared_name.,Adam/audio_classifier/leaf/PCEN/EMA/smooth/v
©
@Adam/audio_classifier/leaf/PCEN/EMA/smooth/v/Read/ReadVariableOpReadVariableOp,Adam/audio_classifier/leaf/PCEN/EMA/smooth/v*
_output_shapes
:(*
dtype0

NoOpNoOp
ý>
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¸>
value®>B«> B¤>

	_frontend
	_pool
	_head
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	
signatures


_complex_conv
_activation
_pooling
_compress_fn
	variables
trainable_variables
regularization_losses
	keras_api
l
layer-0
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
à
iter

beta_1

 beta_2
	!decay
"learning_ratemm#m$m%m&m'm(mvv#v$v%v&v'v (v¡
8
#0
$1
%2
&3
'4
(5
6
7
8
#0
$1
%2
&3
'4
(5
6
7
 
­
)non_trainable_variables

*layers
+metrics
,layer_regularization_losses
-layer_metrics
	variables
trainable_variables
regularization_losses
 
k

#kernel
#_kernel
.	variables
/trainable_variables
0regularization_losses
1	keras_api
]
	2_pool
3	variables
4trainable_variables
5regularization_losses
6	keras_api
^

$kernel
7	variables
8trainable_variables
9regularization_losses
:	keras_api
{
	%alpha
	&delta
'root
;ema
<	variables
=trainable_variables
>regularization_losses
?	keras_api
*
#0
$1
%2
&3
'4
(5
*
#0
$1
%2
&3
'4
(5
 
­
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
	variables
trainable_variables
regularization_losses
R
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
R
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
 
 
 
­
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
trainable_variables
regularization_losses
ZX
VARIABLE_VALUEaudio_classifier/dense/kernel'_head/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEaudio_classifier/dense/bias%_head/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
	variables
trainable_variables
regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
B@
VARIABLE_VALUEkernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE.audio_classifier/leaf/learnable_pooling/kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUE audio_classifier/leaf/PCEN/alpha&variables/2/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUE audio_classifier/leaf/PCEN/delta&variables/3/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEaudio_classifier/leaf/PCEN/root&variables/4/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%audio_classifier/leaf/PCEN/EMA/smooth&variables/5/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2

W0
X1
 
 

#0

#0
 
­
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
.	variables
/trainable_variables
0regularization_losses
R
^	variables
_trainable_variables
`regularization_losses
a	keras_api
 
 
 
­
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
3	variables
4trainable_variables
5regularization_losses

$0

$0
 
­
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
7	variables
8trainable_variables
9regularization_losses
l

(smooth
(_weights
l	variables
mtrainable_variables
nregularization_losses
o	keras_api

%0
&1
'2
(3

%0
&1
'2
(3
 
­
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
<	variables
=trainable_variables
>regularization_losses
 


0
1
2
3
 
 
 
 
 
 
­
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
 
 
 
­
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
 

0
1
 
 
 
 
 
 
 
 
7
	total

count
	variables
	keras_api
I

total

count

_fn_kwargs
	variables
	keras_api
 
 
 
 
 
 
 
 
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
^	variables
_trainable_variables
`regularization_losses
 

20
 
 
 
 
 
 
 
 

(0

(0
 
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
l	variables
mtrainable_variables
nregularization_losses
 

;0
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

0
1

	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

	variables
 
 
 
 
 
 
 
 
 
 
}{
VARIABLE_VALUE$Adam/audio_classifier/dense/kernel/mC_head/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE"Adam/audio_classifier/dense/bias/mA_head/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEAdam/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE5Adam/audio_classifier/leaf/learnable_pooling/kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'Adam/audio_classifier/leaf/PCEN/alpha/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'Adam/audio_classifier/leaf/PCEN/delta/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE&Adam/audio_classifier/leaf/PCEN/root/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/audio_classifier/leaf/PCEN/EMA/smooth/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE$Adam/audio_classifier/dense/kernel/vC_head/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE"Adam/audio_classifier/dense/bias/vA_head/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEAdam/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE5Adam/audio_classifier/leaf/learnable_pooling/kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'Adam/audio_classifier/leaf/PCEN/alpha/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'Adam/audio_classifier/leaf/PCEN/delta/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE&Adam/audio_classifier/leaf/PCEN/root/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/audio_classifier/leaf/PCEN/EMA/smooth/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|
serving_default_input_1Placeholder*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ}
Ä
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1kernel.audio_classifier/leaf/learnable_pooling/kernel audio_classifier/leaf/PCEN/alphaaudio_classifier/leaf/PCEN/root%audio_classifier/leaf/PCEN/EMA/smooth audio_classifier/leaf/PCEN/deltaaudio_classifier/dense/kernelaudio_classifier/dense/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_14249
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
§
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename1audio_classifier/dense/kernel/Read/ReadVariableOp/audio_classifier/dense/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpkernel/Read/ReadVariableOpBaudio_classifier/leaf/learnable_pooling/kernel/Read/ReadVariableOp4audio_classifier/leaf/PCEN/alpha/Read/ReadVariableOp4audio_classifier/leaf/PCEN/delta/Read/ReadVariableOp3audio_classifier/leaf/PCEN/root/Read/ReadVariableOp9audio_classifier/leaf/PCEN/EMA/smooth/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp8Adam/audio_classifier/dense/kernel/m/Read/ReadVariableOp6Adam/audio_classifier/dense/bias/m/Read/ReadVariableOp!Adam/kernel/m/Read/ReadVariableOpIAdam/audio_classifier/leaf/learnable_pooling/kernel/m/Read/ReadVariableOp;Adam/audio_classifier/leaf/PCEN/alpha/m/Read/ReadVariableOp;Adam/audio_classifier/leaf/PCEN/delta/m/Read/ReadVariableOp:Adam/audio_classifier/leaf/PCEN/root/m/Read/ReadVariableOp@Adam/audio_classifier/leaf/PCEN/EMA/smooth/m/Read/ReadVariableOp8Adam/audio_classifier/dense/kernel/v/Read/ReadVariableOp6Adam/audio_classifier/dense/bias/v/Read/ReadVariableOp!Adam/kernel/v/Read/ReadVariableOpIAdam/audio_classifier/leaf/learnable_pooling/kernel/v/Read/ReadVariableOp;Adam/audio_classifier/leaf/PCEN/alpha/v/Read/ReadVariableOp;Adam/audio_classifier/leaf/PCEN/delta/v/Read/ReadVariableOp:Adam/audio_classifier/leaf/PCEN/root/v/Read/ReadVariableOp@Adam/audio_classifier/leaf/PCEN/EMA/smooth/v/Read/ReadVariableOpConst*.
Tin'
%2#	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *'
f"R 
__inference__traced_save_15757


StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameaudio_classifier/dense/kernelaudio_classifier/dense/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratekernel.audio_classifier/leaf/learnable_pooling/kernel audio_classifier/leaf/PCEN/alpha audio_classifier/leaf/PCEN/deltaaudio_classifier/leaf/PCEN/root%audio_classifier/leaf/PCEN/EMA/smoothtotalcounttotal_1count_1$Adam/audio_classifier/dense/kernel/m"Adam/audio_classifier/dense/bias/mAdam/kernel/m5Adam/audio_classifier/leaf/learnable_pooling/kernel/m'Adam/audio_classifier/leaf/PCEN/alpha/m'Adam/audio_classifier/leaf/PCEN/delta/m&Adam/audio_classifier/leaf/PCEN/root/m,Adam/audio_classifier/leaf/PCEN/EMA/smooth/m$Adam/audio_classifier/dense/kernel/v"Adam/audio_classifier/dense/bias/vAdam/kernel/v5Adam/audio_classifier/leaf/learnable_pooling/kernel/v'Adam/audio_classifier/leaf/PCEN/alpha/v'Adam/audio_classifier/leaf/PCEN/delta/v&Adam/audio_classifier/leaf/PCEN/root/v,Adam/audio_classifier/leaf/PCEN/EMA/smooth/v*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_restore_15866Â¢
Ú
h
L__inference_average_pooling1d_layer_call_and_return_conditional_losses_13474

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¯
AvgPoolAvgPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

SqueezeSqueezeAvgPool:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú
h
L__inference_average_pooling1d_layer_call_and_return_conditional_losses_15635

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¯
AvgPoolAvgPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

SqueezeSqueezeAvgPool:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷
ï
4audio_classifier_leaf_PCEN_EMA_scan_while_cond_13383d
`audio_classifier_leaf_pcen_ema_scan_while_audio_classifier_leaf_pcen_ema_scan_while_loop_counterj
faudio_classifier_leaf_pcen_ema_scan_while_audio_classifier_leaf_pcen_ema_scan_while_maximum_iterations9
5audio_classifier_leaf_pcen_ema_scan_while_placeholder;
7audio_classifier_leaf_pcen_ema_scan_while_placeholder_1;
7audio_classifier_leaf_pcen_ema_scan_while_placeholder_2{
waudio_classifier_leaf_pcen_ema_scan_while_audio_classifier_leaf_pcen_ema_scan_while_cond_13383___redundant_placeholder0{
waudio_classifier_leaf_pcen_ema_scan_while_audio_classifier_leaf_pcen_ema_scan_while_cond_13383___redundant_placeholder16
2audio_classifier_leaf_pcen_ema_scan_while_identity
r
0audio_classifier/leaf/PCEN/EMA/scan/while/Less/yConst*
_output_shapes
: *
dtype0*
value	B :dÉ
.audio_classifier/leaf/PCEN/EMA/scan/while/LessLess5audio_classifier_leaf_pcen_ema_scan_while_placeholder9audio_classifier/leaf/PCEN/EMA/scan/while/Less/y:output:0*
T0*
_output_shapes
: £
0audio_classifier/leaf/PCEN/EMA/scan/while/Less_1Less`audio_classifier_leaf_pcen_ema_scan_while_audio_classifier_leaf_pcen_ema_scan_while_loop_counterfaudio_classifier_leaf_pcen_ema_scan_while_audio_classifier_leaf_pcen_ema_scan_while_maximum_iterations*
T0*
_output_shapes
: Ä
4audio_classifier/leaf/PCEN/EMA/scan/while/LogicalAnd
LogicalAnd4audio_classifier/leaf/PCEN/EMA/scan/while/Less_1:z:02audio_classifier/leaf/PCEN/EMA/scan/while/Less:z:0*
_output_shapes
: 
2audio_classifier/leaf/PCEN/EMA/scan/while/IdentityIdentity8audio_classifier/leaf/PCEN/EMA/scan/while/LogicalAnd:z:0*
T0
*
_output_shapes
: "q
2audio_classifier_leaf_pcen_ema_scan_while_identity;audio_classifier/leaf/PCEN/EMA/scan/while/Identity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : :ÿÿÿÿÿÿÿÿÿ(: ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:

_output_shapes
: :

_output_shapes
::

_output_shapes
:
²
F
*__inference_sequential_layer_call_fn_15285

inputs
identity³
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_13993`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd(:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(
 
_user_specified_nameinputs
Æ
P
4__inference_global_max_pooling2d_layer_call_fn_15599

inputs
identity½
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_13955`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd(:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(
 
_user_specified_nameinputs

C
'__inference_flatten_layer_call_fn_15616

inputs
identity°
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_13963`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä

K__inference_audio_classifier_layer_call_and_return_conditional_losses_14053

inputs

leaf_14020:($

leaf_14022:(

leaf_14024:(

leaf_14026:(

leaf_14028:(

leaf_14030:(
dense_14047:
dense_14049:
identity¢dense/StatefulPartitionedCall¢leaf/StatefulPartitionedCall
leaf/StatefulPartitionedCallStatefulPartitionedCallinputs
leaf_14020
leaf_14022
leaf_14024
leaf_14026
leaf_14028
leaf_14030*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_leaf_layer_call_and_return_conditional_losses_13748Y
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ

ExpandDims
ExpandDims%leaf/StatefulPartitionedCall:output:0ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(Ë
sequential/PartitionedCallPartitionedCallExpandDims:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_13966
dense/StatefulPartitionedCallStatefulPartitionedCall#sequential/PartitionedCall:output:0dense_14047dense_14049*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_14046u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense/StatefulPartitionedCall^leaf/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ}: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2<
leaf/StatefulPartitionedCallleaf/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
 
_user_specified_nameinputs
ñ

$__inference_leaf_layer_call_fn_14812

inputs
unknown:(#
	unknown_0:(
	unknown_1:(
	unknown_2:(
	unknown_3:(
	unknown_4:(
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_leaf_layer_call_and_return_conditional_losses_13845s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ}: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
 
_user_specified_nameinputs
½

%__inference_dense_layer_call_fn_15310

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_14046o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü%
ú
#leaf_PCEN_EMA_scan_while_body_14699B
>leaf_pcen_ema_scan_while_leaf_pcen_ema_scan_while_loop_counterH
Dleaf_pcen_ema_scan_while_leaf_pcen_ema_scan_while_maximum_iterations(
$leaf_pcen_ema_scan_while_placeholder*
&leaf_pcen_ema_scan_while_placeholder_1*
&leaf_pcen_ema_scan_while_placeholder_2}
yleaf_pcen_ema_scan_while_tensorarrayv2read_tensorlistgetitem_leaf_pcen_ema_scan_tensorarrayunstack_tensorlistfromtensor_0>
:leaf_pcen_ema_scan_while_mul_leaf_pcen_ema_clip_by_value_0%
!leaf_pcen_ema_scan_while_identity'
#leaf_pcen_ema_scan_while_identity_1'
#leaf_pcen_ema_scan_while_identity_2'
#leaf_pcen_ema_scan_while_identity_3'
#leaf_pcen_ema_scan_while_identity_4{
wleaf_pcen_ema_scan_while_tensorarrayv2read_tensorlistgetitem_leaf_pcen_ema_scan_tensorarrayunstack_tensorlistfromtensor<
8leaf_pcen_ema_scan_while_mul_leaf_pcen_ema_clip_by_value
Jleaf/PCEN/EMA/scan/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   
<leaf/PCEN/EMA/scan/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemyleaf_pcen_ema_scan_while_tensorarrayv2read_tensorlistgetitem_leaf_pcen_ema_scan_tensorarrayunstack_tensorlistfromtensor_0$leaf_pcen_ema_scan_while_placeholderSleaf/PCEN/EMA/scan/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
element_dtype0Ö
leaf/PCEN/EMA/scan/while/mulMul:leaf_pcen_ema_scan_while_mul_leaf_pcen_ema_clip_by_value_0Cleaf/PCEN/EMA/scan/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(c
leaf/PCEN/EMA/scan/while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?­
leaf/PCEN/EMA/scan/while/subSub'leaf/PCEN/EMA/scan/while/sub/x:output:0:leaf_pcen_ema_scan_while_mul_leaf_pcen_ema_clip_by_value_0*
T0*
_output_shapes
:(¡
leaf/PCEN/EMA/scan/while/mul_1Mul leaf/PCEN/EMA/scan/while/sub:z:0&leaf_pcen_ema_scan_while_placeholder_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
leaf/PCEN/EMA/scan/while/addAddV2 leaf/PCEN/EMA/scan/while/mul:z:0"leaf/PCEN/EMA/scan/while/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
=leaf/PCEN/EMA/scan/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem&leaf_pcen_ema_scan_while_placeholder_2$leaf_pcen_ema_scan_while_placeholder leaf/PCEN/EMA/scan/while/add:z:0*
_output_shapes
: *
element_dtype0:éèÒb
 leaf/PCEN/EMA/scan/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
leaf/PCEN/EMA/scan/while/add_1AddV2$leaf_pcen_ema_scan_while_placeholder)leaf/PCEN/EMA/scan/while/add_1/y:output:0*
T0*
_output_shapes
: b
 leaf/PCEN/EMA/scan/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :³
leaf/PCEN/EMA/scan/while/add_2AddV2>leaf_pcen_ema_scan_while_leaf_pcen_ema_scan_while_loop_counter)leaf/PCEN/EMA/scan/while/add_2/y:output:0*
T0*
_output_shapes
: r
!leaf/PCEN/EMA/scan/while/IdentityIdentity"leaf/PCEN/EMA/scan/while/add_2:z:0*
T0*
_output_shapes
: 
#leaf/PCEN/EMA/scan/while/Identity_1IdentityDleaf_pcen_ema_scan_while_leaf_pcen_ema_scan_while_maximum_iterations*
T0*
_output_shapes
: t
#leaf/PCEN/EMA/scan/while/Identity_2Identity"leaf/PCEN/EMA/scan/while/add_1:z:0*
T0*
_output_shapes
: 
#leaf/PCEN/EMA/scan/while/Identity_3Identity leaf/PCEN/EMA/scan/while/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
#leaf/PCEN/EMA/scan/while/Identity_4IdentityMleaf/PCEN/EMA/scan/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "O
!leaf_pcen_ema_scan_while_identity*leaf/PCEN/EMA/scan/while/Identity:output:0"S
#leaf_pcen_ema_scan_while_identity_1,leaf/PCEN/EMA/scan/while/Identity_1:output:0"S
#leaf_pcen_ema_scan_while_identity_2,leaf/PCEN/EMA/scan/while/Identity_2:output:0"S
#leaf_pcen_ema_scan_while_identity_3,leaf/PCEN/EMA/scan/while/Identity_3:output:0"S
#leaf_pcen_ema_scan_while_identity_4,leaf/PCEN/EMA/scan/while/Identity_4:output:0"v
8leaf_pcen_ema_scan_while_mul_leaf_pcen_ema_clip_by_value:leaf_pcen_ema_scan_while_mul_leaf_pcen_ema_clip_by_value_0"ô
wleaf_pcen_ema_scan_while_tensorarrayv2read_tensorlistgetitem_leaf_pcen_ema_scan_tensorarrayunstack_tensorlistfromtensoryleaf_pcen_ema_scan_while_tensorarrayv2read_tensorlistgetitem_leaf_pcen_ema_scan_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : :ÿÿÿÿÿÿÿÿÿ(: : :(: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:(
«
k
O__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_13940

inputs
identityf
Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      m
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô

$__inference_leaf_layer_call_fn_13877
input_1
unknown:(#
	unknown_0:(
	unknown_1:(
	unknown_2:(
	unknown_3:(
	unknown_4:(
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_leaf_layer_call_and_return_conditional_losses_13845s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ}: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
!
_user_specified_name	input_1
±I
¿
__inference__traced_save_15757
file_prefix<
8savev2_audio_classifier_dense_kernel_read_readvariableop:
6savev2_audio_classifier_dense_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop%
!savev2_kernel_read_readvariableopM
Isavev2_audio_classifier_leaf_learnable_pooling_kernel_read_readvariableop?
;savev2_audio_classifier_leaf_pcen_alpha_read_readvariableop?
;savev2_audio_classifier_leaf_pcen_delta_read_readvariableop>
:savev2_audio_classifier_leaf_pcen_root_read_readvariableopD
@savev2_audio_classifier_leaf_pcen_ema_smooth_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopC
?savev2_adam_audio_classifier_dense_kernel_m_read_readvariableopA
=savev2_adam_audio_classifier_dense_bias_m_read_readvariableop,
(savev2_adam_kernel_m_read_readvariableopT
Psavev2_adam_audio_classifier_leaf_learnable_pooling_kernel_m_read_readvariableopF
Bsavev2_adam_audio_classifier_leaf_pcen_alpha_m_read_readvariableopF
Bsavev2_adam_audio_classifier_leaf_pcen_delta_m_read_readvariableopE
Asavev2_adam_audio_classifier_leaf_pcen_root_m_read_readvariableopK
Gsavev2_adam_audio_classifier_leaf_pcen_ema_smooth_m_read_readvariableopC
?savev2_adam_audio_classifier_dense_kernel_v_read_readvariableopA
=savev2_adam_audio_classifier_dense_bias_v_read_readvariableop,
(savev2_adam_kernel_v_read_readvariableopT
Psavev2_adam_audio_classifier_leaf_learnable_pooling_kernel_v_read_readvariableopF
Bsavev2_adam_audio_classifier_leaf_pcen_alpha_v_read_readvariableopF
Bsavev2_adam_audio_classifier_leaf_pcen_delta_v_read_readvariableopE
Asavev2_adam_audio_classifier_leaf_pcen_root_v_read_readvariableopK
Gsavev2_adam_audio_classifier_leaf_pcen_ema_smooth_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ç
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*ð
valueæBã"B'_head/kernel/.ATTRIBUTES/VARIABLE_VALUEB%_head/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBC_head/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBA_head/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBC_head/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBA_head/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH±
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ¡
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:08savev2_audio_classifier_dense_kernel_read_readvariableop6savev2_audio_classifier_dense_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop!savev2_kernel_read_readvariableopIsavev2_audio_classifier_leaf_learnable_pooling_kernel_read_readvariableop;savev2_audio_classifier_leaf_pcen_alpha_read_readvariableop;savev2_audio_classifier_leaf_pcen_delta_read_readvariableop:savev2_audio_classifier_leaf_pcen_root_read_readvariableop@savev2_audio_classifier_leaf_pcen_ema_smooth_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop?savev2_adam_audio_classifier_dense_kernel_m_read_readvariableop=savev2_adam_audio_classifier_dense_bias_m_read_readvariableop(savev2_adam_kernel_m_read_readvariableopPsavev2_adam_audio_classifier_leaf_learnable_pooling_kernel_m_read_readvariableopBsavev2_adam_audio_classifier_leaf_pcen_alpha_m_read_readvariableopBsavev2_adam_audio_classifier_leaf_pcen_delta_m_read_readvariableopAsavev2_adam_audio_classifier_leaf_pcen_root_m_read_readvariableopGsavev2_adam_audio_classifier_leaf_pcen_ema_smooth_m_read_readvariableop?savev2_adam_audio_classifier_dense_kernel_v_read_readvariableop=savev2_adam_audio_classifier_dense_bias_v_read_readvariableop(savev2_adam_kernel_v_read_readvariableopPsavev2_adam_audio_classifier_leaf_learnable_pooling_kernel_v_read_readvariableopBsavev2_adam_audio_classifier_leaf_pcen_alpha_v_read_readvariableopBsavev2_adam_audio_classifier_leaf_pcen_delta_v_read_readvariableopAsavev2_adam_audio_classifier_leaf_pcen_root_v_read_readvariableopGsavev2_adam_audio_classifier_leaf_pcen_ema_smooth_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*÷
_input_shapeså
â: ::: : : : : :(:(:(:(:(:(: : : : :::(:(:(:(:(:(:::(:(:(:(:(:(: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:(:,	(
&
_output_shapes
:(: 


_output_shapes
:(: 

_output_shapes
:(: 

_output_shapes
:(: 

_output_shapes
:(:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:(:,(
&
_output_shapes
:(: 

_output_shapes
:(: 

_output_shapes
:(: 

_output_shapes
:(: 

_output_shapes
:(:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:(:,(
&
_output_shapes
:(: 

_output_shapes
:(: 

_output_shapes
:(:  

_output_shapes
:(: !

_output_shapes
:(:"

_output_shapes
: 
î
Z
*__inference_sequential_layer_call_fn_13969
global_max_pooling2d_input
identityÇ
PartitionedCallPartitionedCallglobal_max_pooling2d_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_13966`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd(:k g
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(
4
_user_specified_nameglobal_max_pooling2d_input
°
¨
EMA_scan_while_body_15523.
*ema_scan_while_ema_scan_while_loop_counter4
0ema_scan_while_ema_scan_while_maximum_iterations
ema_scan_while_placeholder 
ema_scan_while_placeholder_1 
ema_scan_while_placeholder_2i
eema_scan_while_tensorarrayv2read_tensorlistgetitem_ema_scan_tensorarrayunstack_tensorlistfromtensor_0*
&ema_scan_while_mul_ema_clip_by_value_0
ema_scan_while_identity
ema_scan_while_identity_1
ema_scan_while_identity_2
ema_scan_while_identity_3
ema_scan_while_identity_4g
cema_scan_while_tensorarrayv2read_tensorlistgetitem_ema_scan_tensorarrayunstack_tensorlistfromtensor(
$ema_scan_while_mul_ema_clip_by_value
@EMA/scan/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   Ó
2EMA/scan/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemeema_scan_while_tensorarrayv2read_tensorlistgetitem_ema_scan_tensorarrayunstack_tensorlistfromtensor_0ema_scan_while_placeholderIEMA/scan/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
element_dtype0®
EMA/scan/while/mulMul&ema_scan_while_mul_ema_clip_by_value_09EMA/scan/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Y
EMA/scan/while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
EMA/scan/while/subSubEMA/scan/while/sub/x:output:0&ema_scan_while_mul_ema_clip_by_value_0*
T0*
_output_shapes
:(
EMA/scan/while/mul_1MulEMA/scan/while/sub:z:0ema_scan_while_placeholder_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
EMA/scan/while/addAddV2EMA/scan/while/mul:z:0EMA/scan/while/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Ú
3EMA/scan/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemema_scan_while_placeholder_2ema_scan_while_placeholderEMA/scan/while/add:z:0*
_output_shapes
: *
element_dtype0:éèÒX
EMA/scan/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :{
EMA/scan/while/add_1AddV2ema_scan_while_placeholderEMA/scan/while/add_1/y:output:0*
T0*
_output_shapes
: X
EMA/scan/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :
EMA/scan/while/add_2AddV2*ema_scan_while_ema_scan_while_loop_counterEMA/scan/while/add_2/y:output:0*
T0*
_output_shapes
: ^
EMA/scan/while/IdentityIdentityEMA/scan/while/add_2:z:0*
T0*
_output_shapes
: x
EMA/scan/while/Identity_1Identity0ema_scan_while_ema_scan_while_maximum_iterations*
T0*
_output_shapes
: `
EMA/scan/while/Identity_2IdentityEMA/scan/while/add_1:z:0*
T0*
_output_shapes
: o
EMA/scan/while/Identity_3IdentityEMA/scan/while/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
EMA/scan/while/Identity_4IdentityCEMA/scan/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: ";
ema_scan_while_identity EMA/scan/while/Identity:output:0"?
ema_scan_while_identity_1"EMA/scan/while/Identity_1:output:0"?
ema_scan_while_identity_2"EMA/scan/while/Identity_2:output:0"?
ema_scan_while_identity_3"EMA/scan/while/Identity_3:output:0"?
ema_scan_while_identity_4"EMA/scan/while/Identity_4:output:0"N
$ema_scan_while_mul_ema_clip_by_value&ema_scan_while_mul_ema_clip_by_value_0"Ì
cema_scan_while_tensorarrayv2read_tensorlistgetitem_ema_scan_tensorarrayunstack_tensorlistfromtensoreema_scan_while_tensorarrayv2read_tensorlistgetitem_ema_scan_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : :ÿÿÿÿÿÿÿÿÿ(: : :(: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:(
«
k
O__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_15605

inputs
identityf
Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      m
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

a
E__inference_sequential_layer_call_and_return_conditional_losses_15301

inputs
identity{
*global_max_pooling2d/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
global_max_pooling2d/MaxMaxinputs3global_max_pooling2d/Max/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
flatten/ReshapeReshape!global_max_pooling2d/Max:output:0flatten/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentityflatten/Reshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd(:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(
 
_user_specified_nameinputs
M
½
O__inference_tfbanks_complex_conv_layer_call_and_return_conditional_losses_15416

inputs)
readvariableop_resource:(
identity¢ReadVariableOp¢ReadVariableOp_1f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:(*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÿ
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:(*

begin_mask*
end_mask*
shrink_axis_mask\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÛI@
clip_by_value/MinimumMinimumstrided_slice:output:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:(T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    r
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes
:(h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:(*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:(*

begin_mask*
end_mask*
shrink_axis_mask^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *IC
clip_by_value_1/MinimumMinimumstrided_slice_1:output:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:(V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *Tã¿?x
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
:(s
stackPackclip_by_value:z:0clip_by_value_1:z:0*
N*
T0*
_output_shapes

:(*

axisP
range/startConst*
_output_shapes
: *
dtype0*
valueB
 *  HÃP
range/limitConst*
_output_shapes
: *
dtype0*
valueB
 *  ICP
range/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?y
rangeRangerange/start:output:0range/limit:output:0range/delta:output:0*

Tidx0*
_output_shapes	
:f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÿ
strided_slice_2StridedSlicestack:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:(*

begin_mask*
end_mask*
shrink_axis_maskf
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÿ
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:(*

begin_mask*
end_mask*
shrink_axis_maskK
Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÛÉ@>
SqrtSqrtSqrt/x:output:0*
T0*
_output_shapes
: S
mulMulSqrt:y:0strided_slice_3:output:0*
T0*
_output_shapes
:(N
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?T
truedivRealDivtruediv/x:output:0mul:z:0*
T0*
_output_shapes
:(J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
powPowstrided_slice_3:output:0pow/y:output:0*
T0*
_output_shapes
:(L
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @L
mul_1Mulmul_1/x:output:0pow:z:0*
T0*
_output_shapes
:(P
truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
	truediv_1RealDivtruediv_1/x:output:0	mul_1:z:0*
T0*
_output_shapes
:(L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
pow_1Powrange:output:0pow_1/y:output:0*
T0*
_output_shapes	
:;
NegNeg	pow_1:z:0*
T0*
_output_shapes	
:h
Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"(      v
Tensordot/ReshapeReshapetruediv_1:z:0 Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:(j
Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"     u
Tensordot/Reshape_1ReshapeNeg:y:0"Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes
:	~
Tensordot/MatMulMatMulTensordot/Reshape:output:0Tensordot/Reshape_1:output:0*
T0*
_output_shapes
:	(P
ExpExpTensordot/MatMul:product:0*
T0*
_output_shapes
:	(Z
CastCaststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
:(S
Cast_1Castrange:output:0*

DstT0*

SrcT0*
_output_shapes	
:j
Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"(      u
Tensordot_1/ReshapeReshapeCast:y:0"Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes

:(l
Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"     |
Tensordot_1/Reshape_1Reshape
Cast_1:y:0$Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes
:	
Tensordot_1/MatMulMatMulTensordot_1/Reshape:output:0Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes
:	(P
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB J      ?f
mul_2Mulmul_2/x:output:0Tensordot_1/MatMul:product:0*
T0*
_output_shapes
:	(A
Exp_1Exp	mul_2:z:0*
T0*
_output_shapes
:	(O
Cast_2Casttruediv:z:0*

DstT0*

SrcT0*
_output_shapes
:(f
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ü
strided_slice_4StridedSlice
Cast_2:y:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:(*

begin_mask*
end_mask*
new_axis_maskP
Cast_3CastExp:y:0*

DstT0*

SrcT0*
_output_shapes
:	([
mul_3Mulstrided_slice_4:output:0	Exp_1:y:0*
T0*
_output_shapes
:	(M
mul_4Mul	mul_3:z:0
Cast_3:y:0*
T0*
_output_shapes
:	(8
RealReal	mul_4:z:0*
_output_shapes
:	(8
ImagImag	mul_4:z:0*
_output_shapes
:	(p
stack_1PackReal:output:0Imag:output:0*
N*
T0*#
_output_shapes
:(*

axis^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"P     f
ReshapeReshapestack_1:output:0Reshape/shape:output:0*
T0*
_output_shapes
:	P_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       k
	transpose	TransposeReshape:output:0transpose/perm:output:0*
T0*
_output_shapes
:	PP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :n

ExpandDims
ExpandDimstranspose:y:0ExpandDims/dim:output:0*
T0*#
_output_shapes
:P`
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}Y
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
conv1d/ExpandDims_1
ExpandDimsExpandDims:output:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:P­
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}P*
paddingSAME*
strides

conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}P*
squeeze_dims

ýÿÿÿÿÿÿÿÿk
IdentityIdentityconv1d/Squeeze:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}Pj
NoOpNoOp^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:ÿÿÿÿÿÿÿÿÿ}: 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
 
_user_specified_nameinputs
ÿô
Ú
K__inference_audio_classifier_layer_call_and_return_conditional_losses_14534

inputsC
1leaf_tfbanks_complex_conv_readvariableop_resource:(V
<leaf_learnable_pooling_clip_by_value_readvariableop_resource:(7
)leaf_pcen_minimum_readvariableop_resource:(7
)leaf_pcen_maximum_readvariableop_resource:(A
3leaf_pcen_ema_clip_by_value_readvariableop_resource:(5
'leaf_pcen_add_1_readvariableop_resource:(6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢*leaf/PCEN/EMA/clip_by_value/ReadVariableOp¢ leaf/PCEN/Maximum/ReadVariableOp¢ leaf/PCEN/Minimum/ReadVariableOp¢leaf/PCEN/ReadVariableOp¢leaf/PCEN/add_1/ReadVariableOp¢3leaf/learnable_pooling/clip_by_value/ReadVariableOp¢(leaf/tfbanks_complex_conv/ReadVariableOp¢*leaf/tfbanks_complex_conv/ReadVariableOp_1m
leaf/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            o
leaf/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            o
leaf/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
leaf/strided_sliceStridedSliceinputs!leaf/strided_slice/stack:output:0#leaf/strided_slice/stack_1:output:0#leaf/strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*

begin_mask*
end_mask*
new_axis_mask
(leaf/tfbanks_complex_conv/ReadVariableOpReadVariableOp1leaf_tfbanks_complex_conv_readvariableop_resource*
_output_shapes

:(*
dtype0~
-leaf/tfbanks_complex_conv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
/leaf/tfbanks_complex_conv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
/leaf/tfbanks_complex_conv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
'leaf/tfbanks_complex_conv/strided_sliceStridedSlice0leaf/tfbanks_complex_conv/ReadVariableOp:value:06leaf/tfbanks_complex_conv/strided_slice/stack:output:08leaf/tfbanks_complex_conv/strided_slice/stack_1:output:08leaf/tfbanks_complex_conv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:(*

begin_mask*
end_mask*
shrink_axis_maskv
1leaf/tfbanks_complex_conv/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÛI@Í
/leaf/tfbanks_complex_conv/clip_by_value/MinimumMinimum0leaf/tfbanks_complex_conv/strided_slice:output:0:leaf/tfbanks_complex_conv/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:(n
)leaf/tfbanks_complex_conv/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    À
'leaf/tfbanks_complex_conv/clip_by_valueMaximum3leaf/tfbanks_complex_conv/clip_by_value/Minimum:z:02leaf/tfbanks_complex_conv/clip_by_value/y:output:0*
T0*
_output_shapes
:(
*leaf/tfbanks_complex_conv/ReadVariableOp_1ReadVariableOp1leaf_tfbanks_complex_conv_readvariableop_resource*
_output_shapes

:(*
dtype0
/leaf/tfbanks_complex_conv/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       
1leaf/tfbanks_complex_conv/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
1leaf/tfbanks_complex_conv/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
)leaf/tfbanks_complex_conv/strided_slice_1StridedSlice2leaf/tfbanks_complex_conv/ReadVariableOp_1:value:08leaf/tfbanks_complex_conv/strided_slice_1/stack:output:0:leaf/tfbanks_complex_conv/strided_slice_1/stack_1:output:0:leaf/tfbanks_complex_conv/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:(*

begin_mask*
end_mask*
shrink_axis_maskx
3leaf/tfbanks_complex_conv/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ICÓ
1leaf/tfbanks_complex_conv/clip_by_value_1/MinimumMinimum2leaf/tfbanks_complex_conv/strided_slice_1:output:0<leaf/tfbanks_complex_conv/clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:(p
+leaf/tfbanks_complex_conv/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *Tã¿?Æ
)leaf/tfbanks_complex_conv/clip_by_value_1Maximum5leaf/tfbanks_complex_conv/clip_by_value_1/Minimum:z:04leaf/tfbanks_complex_conv/clip_by_value_1/y:output:0*
T0*
_output_shapes
:(Á
leaf/tfbanks_complex_conv/stackPack+leaf/tfbanks_complex_conv/clip_by_value:z:0-leaf/tfbanks_complex_conv/clip_by_value_1:z:0*
N*
T0*
_output_shapes

:(*

axisj
%leaf/tfbanks_complex_conv/range/startConst*
_output_shapes
: *
dtype0*
valueB
 *  HÃj
%leaf/tfbanks_complex_conv/range/limitConst*
_output_shapes
: *
dtype0*
valueB
 *  ICj
%leaf/tfbanks_complex_conv/range/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?á
leaf/tfbanks_complex_conv/rangeRange.leaf/tfbanks_complex_conv/range/start:output:0.leaf/tfbanks_complex_conv/range/limit:output:0.leaf/tfbanks_complex_conv/range/delta:output:0*

Tidx0*
_output_shapes	
:
/leaf/tfbanks_complex_conv/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        
1leaf/tfbanks_complex_conv/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
1leaf/tfbanks_complex_conv/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
)leaf/tfbanks_complex_conv/strided_slice_2StridedSlice(leaf/tfbanks_complex_conv/stack:output:08leaf/tfbanks_complex_conv/strided_slice_2/stack:output:0:leaf/tfbanks_complex_conv/strided_slice_2/stack_1:output:0:leaf/tfbanks_complex_conv/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:(*

begin_mask*
end_mask*
shrink_axis_mask
/leaf/tfbanks_complex_conv/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       
1leaf/tfbanks_complex_conv/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
1leaf/tfbanks_complex_conv/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
)leaf/tfbanks_complex_conv/strided_slice_3StridedSlice(leaf/tfbanks_complex_conv/stack:output:08leaf/tfbanks_complex_conv/strided_slice_3/stack:output:0:leaf/tfbanks_complex_conv/strided_slice_3/stack_1:output:0:leaf/tfbanks_complex_conv/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:(*

begin_mask*
end_mask*
shrink_axis_maske
 leaf/tfbanks_complex_conv/Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÛÉ@r
leaf/tfbanks_complex_conv/SqrtSqrt)leaf/tfbanks_complex_conv/Sqrt/x:output:0*
T0*
_output_shapes
: ¡
leaf/tfbanks_complex_conv/mulMul"leaf/tfbanks_complex_conv/Sqrt:y:02leaf/tfbanks_complex_conv/strided_slice_3:output:0*
T0*
_output_shapes
:(h
#leaf/tfbanks_complex_conv/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¢
!leaf/tfbanks_complex_conv/truedivRealDiv,leaf/tfbanks_complex_conv/truediv/x:output:0!leaf/tfbanks_complex_conv/mul:z:0*
T0*
_output_shapes
:(d
leaf/tfbanks_complex_conv/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @§
leaf/tfbanks_complex_conv/powPow2leaf/tfbanks_complex_conv/strided_slice_3:output:0(leaf/tfbanks_complex_conv/pow/y:output:0*
T0*
_output_shapes
:(f
!leaf/tfbanks_complex_conv/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
leaf/tfbanks_complex_conv/mul_1Mul*leaf/tfbanks_complex_conv/mul_1/x:output:0!leaf/tfbanks_complex_conv/pow:z:0*
T0*
_output_shapes
:(j
%leaf/tfbanks_complex_conv/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¨
#leaf/tfbanks_complex_conv/truediv_1RealDiv.leaf/tfbanks_complex_conv/truediv_1/x:output:0#leaf/tfbanks_complex_conv/mul_1:z:0*
T0*
_output_shapes
:(f
!leaf/tfbanks_complex_conv/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @¢
leaf/tfbanks_complex_conv/pow_1Pow(leaf/tfbanks_complex_conv/range:output:0*leaf/tfbanks_complex_conv/pow_1/y:output:0*
T0*
_output_shapes	
:o
leaf/tfbanks_complex_conv/NegNeg#leaf/tfbanks_complex_conv/pow_1:z:0*
T0*
_output_shapes	
:
1leaf/tfbanks_complex_conv/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"(      Ä
+leaf/tfbanks_complex_conv/Tensordot/ReshapeReshape'leaf/tfbanks_complex_conv/truediv_1:z:0:leaf/tfbanks_complex_conv/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:(
3leaf/tfbanks_complex_conv/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"     Ã
-leaf/tfbanks_complex_conv/Tensordot/Reshape_1Reshape!leaf/tfbanks_complex_conv/Neg:y:0<leaf/tfbanks_complex_conv/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes
:	Ì
*leaf/tfbanks_complex_conv/Tensordot/MatMulMatMul4leaf/tfbanks_complex_conv/Tensordot/Reshape:output:06leaf/tfbanks_complex_conv/Tensordot/Reshape_1:output:0*
T0*
_output_shapes
:	(
leaf/tfbanks_complex_conv/ExpExp4leaf/tfbanks_complex_conv/Tensordot/MatMul:product:0*
T0*
_output_shapes
:	(
leaf/tfbanks_complex_conv/CastCast2leaf/tfbanks_complex_conv/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
:(
 leaf/tfbanks_complex_conv/Cast_1Cast(leaf/tfbanks_complex_conv/range:output:0*

DstT0*

SrcT0*
_output_shapes	
:
3leaf/tfbanks_complex_conv/Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"(      Ã
-leaf/tfbanks_complex_conv/Tensordot_1/ReshapeReshape"leaf/tfbanks_complex_conv/Cast:y:0<leaf/tfbanks_complex_conv/Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes

:(
5leaf/tfbanks_complex_conv/Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"     Ê
/leaf/tfbanks_complex_conv/Tensordot_1/Reshape_1Reshape$leaf/tfbanks_complex_conv/Cast_1:y:0>leaf/tfbanks_complex_conv/Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes
:	Ò
,leaf/tfbanks_complex_conv/Tensordot_1/MatMulMatMul6leaf/tfbanks_complex_conv/Tensordot_1/Reshape:output:08leaf/tfbanks_complex_conv/Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes
:	(j
!leaf/tfbanks_complex_conv/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB J      ?´
leaf/tfbanks_complex_conv/mul_2Mul*leaf/tfbanks_complex_conv/mul_2/x:output:06leaf/tfbanks_complex_conv/Tensordot_1/MatMul:product:0*
T0*
_output_shapes
:	(u
leaf/tfbanks_complex_conv/Exp_1Exp#leaf/tfbanks_complex_conv/mul_2:z:0*
T0*
_output_shapes
:	(
 leaf/tfbanks_complex_conv/Cast_2Cast%leaf/tfbanks_complex_conv/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
:(
/leaf/tfbanks_complex_conv/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        
1leaf/tfbanks_complex_conv/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
1leaf/tfbanks_complex_conv/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      þ
)leaf/tfbanks_complex_conv/strided_slice_4StridedSlice$leaf/tfbanks_complex_conv/Cast_2:y:08leaf/tfbanks_complex_conv/strided_slice_4/stack:output:0:leaf/tfbanks_complex_conv/strided_slice_4/stack_1:output:0:leaf/tfbanks_complex_conv/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:(*

begin_mask*
end_mask*
new_axis_mask
 leaf/tfbanks_complex_conv/Cast_3Cast!leaf/tfbanks_complex_conv/Exp:y:0*

DstT0*

SrcT0*
_output_shapes
:	(©
leaf/tfbanks_complex_conv/mul_3Mul2leaf/tfbanks_complex_conv/strided_slice_4:output:0#leaf/tfbanks_complex_conv/Exp_1:y:0*
T0*
_output_shapes
:	(
leaf/tfbanks_complex_conv/mul_4Mul#leaf/tfbanks_complex_conv/mul_3:z:0$leaf/tfbanks_complex_conv/Cast_3:y:0*
T0*
_output_shapes
:	(l
leaf/tfbanks_complex_conv/RealReal#leaf/tfbanks_complex_conv/mul_4:z:0*
_output_shapes
:	(l
leaf/tfbanks_complex_conv/ImagImag#leaf/tfbanks_complex_conv/mul_4:z:0*
_output_shapes
:	(¾
!leaf/tfbanks_complex_conv/stack_1Pack'leaf/tfbanks_complex_conv/Real:output:0'leaf/tfbanks_complex_conv/Imag:output:0*
N*
T0*#
_output_shapes
:(*

axisx
'leaf/tfbanks_complex_conv/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"P     ´
!leaf/tfbanks_complex_conv/ReshapeReshape*leaf/tfbanks_complex_conv/stack_1:output:00leaf/tfbanks_complex_conv/Reshape/shape:output:0*
T0*
_output_shapes
:	Py
(leaf/tfbanks_complex_conv/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       ¹
#leaf/tfbanks_complex_conv/transpose	Transpose*leaf/tfbanks_complex_conv/Reshape:output:01leaf/tfbanks_complex_conv/transpose/perm:output:0*
T0*
_output_shapes
:	Pj
(leaf/tfbanks_complex_conv/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :¼
$leaf/tfbanks_complex_conv/ExpandDims
ExpandDims'leaf/tfbanks_complex_conv/transpose:y:01leaf/tfbanks_complex_conv/ExpandDims/dim:output:0*
T0*#
_output_shapes
:Pz
/leaf/tfbanks_complex_conv/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿË
+leaf/tfbanks_complex_conv/conv1d/ExpandDims
ExpandDimsleaf/strided_slice:output:08leaf/tfbanks_complex_conv/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}s
1leaf/tfbanks_complex_conv/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Ø
-leaf/tfbanks_complex_conv/conv1d/ExpandDims_1
ExpandDims-leaf/tfbanks_complex_conv/ExpandDims:output:0:leaf/tfbanks_complex_conv/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:Pû
 leaf/tfbanks_complex_conv/conv1dConv2D4leaf/tfbanks_complex_conv/conv1d/ExpandDims:output:06leaf/tfbanks_complex_conv/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}P*
paddingSAME*
strides
µ
(leaf/tfbanks_complex_conv/conv1d/SqueezeSqueeze)leaf/tfbanks_complex_conv/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}P*
squeeze_dims

ýÿÿÿÿÿÿÿÿx
#leaf/squared_modulus/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ã
leaf/squared_modulus/transpose	Transpose1leaf/tfbanks_complex_conv/conv1d/Squeeze:output:0,leaf/squared_modulus/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}_
leaf/squared_modulus/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
leaf/squared_modulus/powPow"leaf/squared_modulus/transpose:y:0#leaf/squared_modulus/pow/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}w
5leaf/squared_modulus/average_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Ø
1leaf/squared_modulus/average_pooling1d/ExpandDims
ExpandDimsleaf/squared_modulus/pow:z:0>leaf/squared_modulus/average_pooling1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}ì
.leaf/squared_modulus/average_pooling1d/AvgPoolAvgPool:leaf/squared_modulus/average_pooling1d/ExpandDims:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}*
ksize
*
paddingVALID*
strides
À
.leaf/squared_modulus/average_pooling1d/SqueezeSqueeze7leaf/squared_modulus/average_pooling1d/AvgPool:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}*
squeeze_dims
_
leaf/squared_modulus/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @´
leaf/squared_modulus/mulMul#leaf/squared_modulus/mul/x:output:07leaf/squared_modulus/average_pooling1d/Squeeze:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}z
%leaf/squared_modulus/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ²
 leaf/squared_modulus/transpose_1	Transposeleaf/squared_modulus/mul:z:0.leaf/squared_modulus/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}(¸
3leaf/learnable_pooling/clip_by_value/ReadVariableOpReadVariableOp<leaf_learnable_pooling_clip_by_value_readvariableop_resource*&
_output_shapes
:(*
dtype0s
.leaf/learnable_pooling/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Þ
,leaf/learnable_pooling/clip_by_value/MinimumMinimum;leaf/learnable_pooling/clip_by_value/ReadVariableOp:value:07leaf/learnable_pooling/clip_by_value/Minimum/y:output:0*
T0*&
_output_shapes
:(k
&leaf/learnable_pooling/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *rn£;Ã
$leaf/learnable_pooling/clip_by_valueMaximum0leaf/learnable_pooling/clip_by_value/Minimum:z:0/leaf/learnable_pooling/clip_by_value/y:output:0*
T0*&
_output_shapes
:(g
"leaf/learnable_pooling/range/startConst*
_output_shapes
: *
dtype0*
valueB
 *    g
"leaf/learnable_pooling/range/limitConst*
_output_shapes
: *
dtype0*
valueB
 * ÈCg
"leaf/learnable_pooling/range/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Õ
leaf/learnable_pooling/rangeRange+leaf/learnable_pooling/range/start:output:0+leaf/learnable_pooling/range/limit:output:0+leaf/learnable_pooling/range/delta:output:0*

Tidx0*
_output_shapes	
:}
$leaf/learnable_pooling/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"           ±
leaf/learnable_pooling/ReshapeReshape%leaf/learnable_pooling/range:output:0-leaf/learnable_pooling/Reshape/shape:output:0*
T0*'
_output_shapes
:a
leaf/learnable_pooling/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HC£
leaf/learnable_pooling/subSub'leaf/learnable_pooling/Reshape:output:0%leaf/learnable_pooling/sub/y:output:0*
T0*'
_output_shapes
:a
leaf/learnable_pooling/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?£
leaf/learnable_pooling/mulMul(leaf/learnable_pooling/clip_by_value:z:0%leaf/learnable_pooling/mul/y:output:0*
T0*&
_output_shapes
:(c
leaf/learnable_pooling/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ÈC
leaf/learnable_pooling/mul_1Mulleaf/learnable_pooling/mul:z:0'leaf/learnable_pooling/mul_1/y:output:0*
T0*&
_output_shapes
:(
leaf/learnable_pooling/truedivRealDivleaf/learnable_pooling/sub:z:0 leaf/learnable_pooling/mul_1:z:0*
T0*'
_output_shapes
:(a
leaf/learnable_pooling/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
leaf/learnable_pooling/powPow"leaf/learnable_pooling/truediv:z:0%leaf/learnable_pooling/pow/y:output:0*
T0*'
_output_shapes
:(c
leaf/learnable_pooling/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ¿
leaf/learnable_pooling/mul_2Mul'leaf/learnable_pooling/mul_2/x:output:0leaf/learnable_pooling/pow:z:0*
T0*'
_output_shapes
:(u
leaf/learnable_pooling/ExpExp leaf/learnable_pooling/mul_2:z:0*
T0*'
_output_shapes
:(g
%leaf/learnable_pooling/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :À
!leaf/learnable_pooling/ExpandDims
ExpandDims$leaf/squared_modulus/transpose_1:y:0.leaf/learnable_pooling/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}(
&leaf/learnable_pooling/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"     (      
.leaf/learnable_pooling/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      é
 leaf/learnable_pooling/depthwiseDepthwiseConv2dNative*leaf/learnable_pooling/ExpandDims:output:0leaf/learnable_pooling/Exp:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(*
paddingSAME*
strides

  ¡
leaf/learnable_pooling/SqueezeSqueeze)leaf/learnable_pooling/depthwise:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(*
squeeze_dims
S
leaf/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7
leaf/MaximumMaximum'leaf/learnable_pooling/Squeeze:output:0leaf/Maximum/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(
 leaf/PCEN/Minimum/ReadVariableOpReadVariableOp)leaf_pcen_minimum_readvariableop_resource*
_output_shapes
:(*
dtype0X
leaf/PCEN/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
leaf/PCEN/MinimumMinimum(leaf/PCEN/Minimum/ReadVariableOp:value:0leaf/PCEN/Minimum/y:output:0*
T0*
_output_shapes
:(
 leaf/PCEN/Maximum/ReadVariableOpReadVariableOp)leaf_pcen_maximum_readvariableop_resource*
_output_shapes
:(*
dtype0X
leaf/PCEN/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
leaf/PCEN/MaximumMaximum(leaf/PCEN/Maximum/ReadVariableOp:value:0leaf/PCEN/Maximum/y:output:0*
T0*
_output_shapes
:(\
leaf/PCEN/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B : Y
leaf/PCEN/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :Ì
leaf/PCEN/GatherV2GatherV2leaf/Maximum:z:0#leaf/PCEN/GatherV2/indices:output:0 leaf/PCEN/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
*leaf/PCEN/EMA/clip_by_value/ReadVariableOpReadVariableOp3leaf_pcen_ema_clip_by_value_readvariableop_resource*
_output_shapes
:(*
dtype0j
%leaf/PCEN/EMA/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?·
#leaf/PCEN/EMA/clip_by_value/MinimumMinimum2leaf/PCEN/EMA/clip_by_value/ReadVariableOp:value:0.leaf/PCEN/EMA/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:(b
leaf/PCEN/EMA/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
leaf/PCEN/EMA/clip_by_valueMaximum'leaf/PCEN/EMA/clip_by_value/Minimum:z:0&leaf/PCEN/EMA/clip_by_value/y:output:0*
T0*
_output_shapes
:(q
leaf/PCEN/EMA/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
leaf/PCEN/EMA/transpose	Transposeleaf/Maximum:z:0%leaf/PCEN/EMA/transpose/perm:output:0*
T0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿ(
.leaf/PCEN/EMA/scan/TensorArrayV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   o
-leaf/PCEN/EMA/scan/TensorArrayV2/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :dø
 leaf/PCEN/EMA/scan/TensorArrayV2TensorListReserve7leaf/PCEN/EMA/scan/TensorArrayV2/element_shape:output:06leaf/PCEN/EMA/scan/TensorArrayV2/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Hleaf/PCEN/EMA/scan/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   
:leaf/PCEN/EMA/scan/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorleaf/PCEN/EMA/transpose:y:0Qleaf/PCEN/EMA/scan/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
0leaf/PCEN/EMA/scan/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   q
/leaf/PCEN/EMA/scan/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :dþ
"leaf/PCEN/EMA/scan/TensorArrayV2_1TensorListReserve9leaf/PCEN/EMA/scan/TensorArrayV2_1/element_shape:output:08leaf/PCEN/EMA/scan/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒZ
leaf/PCEN/EMA/scan/ConstConst*
_output_shapes
: *
dtype0*
value	B : m
+leaf/PCEN/EMA/scan/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
value	B :dg
%leaf/PCEN/EMA/scan/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ì
leaf/PCEN/EMA/scan/whileStatelessWhile.leaf/PCEN/EMA/scan/while/loop_counter:output:04leaf/PCEN/EMA/scan/while/maximum_iterations:output:0!leaf/PCEN/EMA/scan/Const:output:0leaf/PCEN/GatherV2:output:0+leaf/PCEN/EMA/scan/TensorArrayV2_1:handle:0Jleaf/PCEN/EMA/scan/TensorArrayUnstack/TensorListFromTensor:output_handle:0leaf/PCEN/EMA/clip_by_value:z:0*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*7
_output_shapes%
#: : : :ÿÿÿÿÿÿÿÿÿ(: : :(* 
_read_only_resource_inputs
 *
_stateful_parallelism( */
body'R%
#leaf_PCEN_EMA_scan_while_body_14456*/
cond'R%
#leaf_PCEN_EMA_scan_while_cond_14455*6
output_shapes%
#: : : :ÿÿÿÿÿÿÿÿÿ(: : :(
Cleaf/PCEN/EMA/scan/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   
5leaf/PCEN/EMA/scan/TensorArrayV2Stack/TensorListStackTensorListStack!leaf/PCEN/EMA/scan/while:output:4Lleaf/PCEN/EMA/scan/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿ(*
element_dtype0*
num_elementsds
leaf/PCEN/EMA/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Å
leaf/PCEN/EMA/transpose_1	Transpose>leaf/PCEN/EMA/scan/TensorArrayV2Stack/TensorListStack:tensor:0'leaf/PCEN/EMA/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(X
leaf/PCEN/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?v
leaf/PCEN/truedivRealDivleaf/PCEN/truediv/x:output:0leaf/PCEN/Maximum:z:0*
T0*
_output_shapes
:(T
leaf/PCEN/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ì¼+
leaf/PCEN/addAddV2leaf/PCEN/add/x:output:0leaf/PCEN/EMA/transpose_1:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(t
leaf/PCEN/powPowleaf/PCEN/add:z:0leaf/PCEN/Minimum:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(y
leaf/PCEN/truediv_1RealDivleaf/Maximum:z:0leaf/PCEN/pow:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(
leaf/PCEN/add_1/ReadVariableOpReadVariableOp'leaf_pcen_add_1_readvariableop_resource*
_output_shapes
:(*
dtype0
leaf/PCEN/add_1AddV2leaf/PCEN/truediv_1:z:0&leaf/PCEN/add_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(x
leaf/PCEN/pow_1Powleaf/PCEN/add_1:z:0leaf/PCEN/truediv:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(|
leaf/PCEN/ReadVariableOpReadVariableOp'leaf_pcen_add_1_readvariableop_resource*
_output_shapes
:(*
dtype0t
leaf/PCEN/pow_2Pow leaf/PCEN/ReadVariableOp:value:0leaf/PCEN/truediv:z:0*
T0*
_output_shapes
:(t
leaf/PCEN/subSubleaf/PCEN/pow_1:z:0leaf/PCEN/pow_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(Y
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ~

ExpandDims
ExpandDimsleaf/PCEN/sub:z:0ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(
5sequential/global_max_pooling2d/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ±
#sequential/global_max_pooling2d/MaxMaxExpandDims:output:0>sequential/global_max_pooling2d/Max/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¨
sequential/flatten/ReshapeReshape,sequential/global_max_pooling2d/Max:output:0!sequential/flatten/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense/MatMulMatMul#sequential/flatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp+^leaf/PCEN/EMA/clip_by_value/ReadVariableOp!^leaf/PCEN/Maximum/ReadVariableOp!^leaf/PCEN/Minimum/ReadVariableOp^leaf/PCEN/ReadVariableOp^leaf/PCEN/add_1/ReadVariableOp4^leaf/learnable_pooling/clip_by_value/ReadVariableOp)^leaf/tfbanks_complex_conv/ReadVariableOp+^leaf/tfbanks_complex_conv/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ}: : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2X
*leaf/PCEN/EMA/clip_by_value/ReadVariableOp*leaf/PCEN/EMA/clip_by_value/ReadVariableOp2D
 leaf/PCEN/Maximum/ReadVariableOp leaf/PCEN/Maximum/ReadVariableOp2D
 leaf/PCEN/Minimum/ReadVariableOp leaf/PCEN/Minimum/ReadVariableOp24
leaf/PCEN/ReadVariableOpleaf/PCEN/ReadVariableOp2@
leaf/PCEN/add_1/ReadVariableOpleaf/PCEN/add_1/ReadVariableOp2j
3leaf/learnable_pooling/clip_by_value/ReadVariableOp3leaf/learnable_pooling/clip_by_value/ReadVariableOp2T
(leaf/tfbanks_complex_conv/ReadVariableOp(leaf/tfbanks_complex_conv/ReadVariableOp2X
*leaf/tfbanks_complex_conv/ReadVariableOp_1*leaf/tfbanks_complex_conv/ReadVariableOp_1:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
 
_user_specified_nameinputs
±
F
/__inference_squared_modulus_layer_call_fn_15421
x
identity¸
PartitionedCallPartitionedCallx*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_squared_modulus_layer_call_and_return_conditional_losses_13597e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ}P:O K
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}P

_user_specified_namex
6

?__inference_PCEN_layer_call_and_return_conditional_losses_13737

inputs-
minimum_readvariableop_resource:(-
maximum_readvariableop_resource:(7
)ema_clip_by_value_readvariableop_resource:(+
add_1_readvariableop_resource:(
identity¢ EMA/clip_by_value/ReadVariableOp¢Maximum/ReadVariableOp¢Minimum/ReadVariableOp¢ReadVariableOp¢add_1/ReadVariableOpr
Minimum/ReadVariableOpReadVariableOpminimum_readvariableop_resource*
_output_shapes
:(*
dtype0N
	Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?k
MinimumMinimumMinimum/ReadVariableOp:value:0Minimum/y:output:0*
T0*
_output_shapes
:(r
Maximum/ReadVariableOpReadVariableOpmaximum_readvariableop_resource*
_output_shapes
:(*
dtype0N
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?k
MaximumMaximumMaximum/ReadVariableOp:value:0Maximum/y:output:0*
T0*
_output_shapes
:(R
GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B : O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :¤
GatherV2GatherV2inputsGatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 EMA/clip_by_value/ReadVariableOpReadVariableOp)ema_clip_by_value_readvariableop_resource*
_output_shapes
:(*
dtype0`
EMA/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
EMA/clip_by_value/MinimumMinimum(EMA/clip_by_value/ReadVariableOp:value:0$EMA/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:(X
EMA/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
EMA/clip_by_valueMaximumEMA/clip_by_value/Minimum:z:0EMA/clip_by_value/y:output:0*
T0*
_output_shapes
:(g
EMA/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          u
EMA/transpose	TransposeinputsEMA/transpose/perm:output:0*
T0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿ(u
$EMA/scan/TensorArrayV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   e
#EMA/scan/TensorArrayV2/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :dÚ
EMA/scan/TensorArrayV2TensorListReserve-EMA/scan/TensorArrayV2/element_shape:output:0,EMA/scan/TensorArrayV2/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
>EMA/scan/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   ö
0EMA/scan/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorEMA/transpose:y:0GEMA/scan/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒw
&EMA/scan/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   g
%EMA/scan/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :dà
EMA/scan/TensorArrayV2_1TensorListReserve/EMA/scan/TensorArrayV2_1/element_shape:output:0.EMA/scan/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒP
EMA/scan/ConstConst*
_output_shapes
: *
dtype0*
value	B : c
!EMA/scan/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
value	B :d]
EMA/scan/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : è
EMA/scan/whileStatelessWhile$EMA/scan/while/loop_counter:output:0*EMA/scan/while/maximum_iterations:output:0EMA/scan/Const:output:0GatherV2:output:0!EMA/scan/TensorArrayV2_1:handle:0@EMA/scan/TensorArrayUnstack/TensorListFromTensor:output_handle:0EMA/clip_by_value:z:0*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*7
_output_shapes%
#: : : :ÿÿÿÿÿÿÿÿÿ(: : :(* 
_read_only_resource_inputs
 *
_stateful_parallelism( *%
bodyR
EMA_scan_while_body_13671*%
condR
EMA_scan_while_cond_13670*6
output_shapes%
#: : : :ÿÿÿÿÿÿÿÿÿ(: : :(
9EMA/scan/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   ñ
+EMA/scan/TensorArrayV2Stack/TensorListStackTensorListStackEMA/scan/while:output:4BEMA/scan/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿ(*
element_dtype0*
num_elementsdi
EMA/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          §
EMA/transpose_1	Transpose4EMA/scan/TensorArrayV2Stack/TensorListStack:tensor:0EMA/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(N
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?X
truedivRealDivtruediv/x:output:0Maximum:z:0*
T0*
_output_shapes
:(J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ì¼+g
addAddV2add/x:output:0EMA/transpose_1:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(V
powPowadd:z:0Minimum:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd([
	truediv_1RealDivinputspow:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(n
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:(*
dtype0q
add_1AddV2truediv_1:z:0add_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(Z
pow_1Pow	add_1:z:0truediv:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(h
ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:(*
dtype0V
pow_2PowReadVariableOp:value:0truediv:z:0*
T0*
_output_shapes
:(V
subSub	pow_1:z:0	pow_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(Z
IdentityIdentitysub:z:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(Ã
NoOpNoOp!^EMA/clip_by_value/ReadVariableOp^Maximum/ReadVariableOp^Minimum/ReadVariableOp^ReadVariableOp^add_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿd(: : : : 2D
 EMA/clip_by_value/ReadVariableOp EMA/clip_by_value/ReadVariableOp20
Maximum/ReadVariableOpMaximum/ReadVariableOp20
Minimum/ReadVariableOpMinimum/ReadVariableOp2 
ReadVariableOpReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(
 
_user_specified_nameinputs


#leaf_PCEN_EMA_scan_while_cond_14698B
>leaf_pcen_ema_scan_while_leaf_pcen_ema_scan_while_loop_counterH
Dleaf_pcen_ema_scan_while_leaf_pcen_ema_scan_while_maximum_iterations(
$leaf_pcen_ema_scan_while_placeholder*
&leaf_pcen_ema_scan_while_placeholder_1*
&leaf_pcen_ema_scan_while_placeholder_2Y
Uleaf_pcen_ema_scan_while_leaf_pcen_ema_scan_while_cond_14698___redundant_placeholder0Y
Uleaf_pcen_ema_scan_while_leaf_pcen_ema_scan_while_cond_14698___redundant_placeholder1%
!leaf_pcen_ema_scan_while_identity
a
leaf/PCEN/EMA/scan/while/Less/yConst*
_output_shapes
: *
dtype0*
value	B :d
leaf/PCEN/EMA/scan/while/LessLess$leaf_pcen_ema_scan_while_placeholder(leaf/PCEN/EMA/scan/while/Less/y:output:0*
T0*
_output_shapes
: Î
leaf/PCEN/EMA/scan/while/Less_1Less>leaf_pcen_ema_scan_while_leaf_pcen_ema_scan_while_loop_counterDleaf_pcen_ema_scan_while_leaf_pcen_ema_scan_while_maximum_iterations*
T0*
_output_shapes
: 
#leaf/PCEN/EMA/scan/while/LogicalAnd
LogicalAnd#leaf/PCEN/EMA/scan/while/Less_1:z:0!leaf/PCEN/EMA/scan/while/Less:z:0*
_output_shapes
: w
!leaf/PCEN/EMA/scan/while/IdentityIdentity'leaf/PCEN/EMA/scan/while/LogicalAnd:z:0*
T0
*
_output_shapes
: "O
!leaf_pcen_ema_scan_while_identity*leaf/PCEN/EMA/scan/while/Identity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : :ÿÿÿÿÿÿÿÿÿ(: ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:

_output_shapes
: :

_output_shapes
::

_output_shapes
:
Ç

K__inference_audio_classifier_layer_call_and_return_conditional_losses_14195
input_1

leaf_14173:($

leaf_14175:(

leaf_14177:(

leaf_14179:(

leaf_14181:(

leaf_14183:(
dense_14189:
dense_14191:
identity¢dense/StatefulPartitionedCall¢leaf/StatefulPartitionedCall
leaf/StatefulPartitionedCallStatefulPartitionedCallinput_1
leaf_14173
leaf_14175
leaf_14177
leaf_14179
leaf_14181
leaf_14183*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_leaf_layer_call_and_return_conditional_losses_13748Y
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ

ExpandDims
ExpandDims%leaf/StatefulPartitionedCall:output:0ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(Ë
sequential/PartitionedCallPartitionedCallExpandDims:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_13966
dense/StatefulPartitionedCallStatefulPartitionedCall#sequential/PartitionedCall:output:0dense_14189dense_14191*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_14046u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense/StatefulPartitionedCall^leaf/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ}: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2<
leaf/StatefulPartitionedCallleaf/StatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
!
_user_specified_name	input_1
²
F
*__inference_sequential_layer_call_fn_15280

inputs
identity³
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_13966`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd(:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(
 
_user_specified_nameinputs
M
½
O__inference_tfbanks_complex_conv_layer_call_and_return_conditional_losses_13577

inputs)
readvariableop_resource:(
identity¢ReadVariableOp¢ReadVariableOp_1f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:(*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÿ
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:(*

begin_mask*
end_mask*
shrink_axis_mask\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÛI@
clip_by_value/MinimumMinimumstrided_slice:output:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:(T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    r
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes
:(h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:(*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:(*

begin_mask*
end_mask*
shrink_axis_mask^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *IC
clip_by_value_1/MinimumMinimumstrided_slice_1:output:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:(V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *Tã¿?x
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes
:(s
stackPackclip_by_value:z:0clip_by_value_1:z:0*
N*
T0*
_output_shapes

:(*

axisP
range/startConst*
_output_shapes
: *
dtype0*
valueB
 *  HÃP
range/limitConst*
_output_shapes
: *
dtype0*
valueB
 *  ICP
range/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?y
rangeRangerange/start:output:0range/limit:output:0range/delta:output:0*

Tidx0*
_output_shapes	
:f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÿ
strided_slice_2StridedSlicestack:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:(*

begin_mask*
end_mask*
shrink_axis_maskf
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÿ
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:(*

begin_mask*
end_mask*
shrink_axis_maskK
Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÛÉ@>
SqrtSqrtSqrt/x:output:0*
T0*
_output_shapes
: S
mulMulSqrt:y:0strided_slice_3:output:0*
T0*
_output_shapes
:(N
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?T
truedivRealDivtruediv/x:output:0mul:z:0*
T0*
_output_shapes
:(J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
powPowstrided_slice_3:output:0pow/y:output:0*
T0*
_output_shapes
:(L
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @L
mul_1Mulmul_1/x:output:0pow:z:0*
T0*
_output_shapes
:(P
truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Z
	truediv_1RealDivtruediv_1/x:output:0	mul_1:z:0*
T0*
_output_shapes
:(L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @T
pow_1Powrange:output:0pow_1/y:output:0*
T0*
_output_shapes	
:;
NegNeg	pow_1:z:0*
T0*
_output_shapes	
:h
Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"(      v
Tensordot/ReshapeReshapetruediv_1:z:0 Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:(j
Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"     u
Tensordot/Reshape_1ReshapeNeg:y:0"Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes
:	~
Tensordot/MatMulMatMulTensordot/Reshape:output:0Tensordot/Reshape_1:output:0*
T0*
_output_shapes
:	(P
ExpExpTensordot/MatMul:product:0*
T0*
_output_shapes
:	(Z
CastCaststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
:(S
Cast_1Castrange:output:0*

DstT0*

SrcT0*
_output_shapes	
:j
Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"(      u
Tensordot_1/ReshapeReshapeCast:y:0"Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes

:(l
Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"     |
Tensordot_1/Reshape_1Reshape
Cast_1:y:0$Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes
:	
Tensordot_1/MatMulMatMulTensordot_1/Reshape:output:0Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes
:	(P
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB J      ?f
mul_2Mulmul_2/x:output:0Tensordot_1/MatMul:product:0*
T0*
_output_shapes
:	(A
Exp_1Exp	mul_2:z:0*
T0*
_output_shapes
:	(O
Cast_2Casttruediv:z:0*

DstT0*

SrcT0*
_output_shapes
:(f
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ü
strided_slice_4StridedSlice
Cast_2:y:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:(*

begin_mask*
end_mask*
new_axis_maskP
Cast_3CastExp:y:0*

DstT0*

SrcT0*
_output_shapes
:	([
mul_3Mulstrided_slice_4:output:0	Exp_1:y:0*
T0*
_output_shapes
:	(M
mul_4Mul	mul_3:z:0
Cast_3:y:0*
T0*
_output_shapes
:	(8
RealReal	mul_4:z:0*
_output_shapes
:	(8
ImagImag	mul_4:z:0*
_output_shapes
:	(p
stack_1PackReal:output:0Imag:output:0*
N*
T0*#
_output_shapes
:(*

axis^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"P     f
ReshapeReshapestack_1:output:0Reshape/shape:output:0*
T0*
_output_shapes
:	P_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       k
	transpose	TransposeReshape:output:0transpose/perm:output:0*
T0*
_output_shapes
:	PP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :n

ExpandDims
ExpandDimstranspose:y:0ExpandDims/dim:output:0*
T0*#
_output_shapes
:P`
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}Y
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
conv1d/ExpandDims_1
ExpandDimsExpandDims:output:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:P­
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}P*
paddingSAME*
strides

conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}P*
squeeze_dims

ýÿÿÿÿÿÿÿÿk
IdentityIdentityconv1d/Squeeze:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}Pj
NoOpNoOp^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:ÿÿÿÿÿÿÿÿÿ}: 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
 
_user_specified_nameinputs
ã
k
O__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_15611

inputs
identityf
Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      d
MaxMaxinputsMax/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
IdentityIdentityMax:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd(:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(
 
_user_specified_nameinputs
õ
Ë
L__inference_learnable_pooling_layer_call_and_return_conditional_losses_15478

inputs?
%clip_by_value_readvariableop_resource:(
identity¢clip_by_value/ReadVariableOp
clip_by_value/ReadVariableOpReadVariableOp%clip_by_value_readvariableop_resource*&
_output_shapes
:(*
dtype0\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
clip_by_value/MinimumMinimum$clip_by_value/ReadVariableOp:value:0 clip_by_value/Minimum/y:output:0*
T0*&
_output_shapes
:(T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *rn£;~
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*&
_output_shapes
:(P
range/startConst*
_output_shapes
: *
dtype0*
valueB
 *    P
range/limitConst*
_output_shapes
: *
dtype0*
valueB
 * ÈCP
range/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?y
rangeRangerange/start:output:0range/limit:output:0range/delta:output:0*

Tidx0*
_output_shapes	
:f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"           l
ReshapeReshaperange:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HC^
subSubReshape:output:0sub/y:output:0*
T0*'
_output_shapes
:J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?^
mulMulclip_by_value:z:0mul/y:output:0*
T0*&
_output_shapes
:(L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ÈCX
mul_1Mulmul:z:0mul_1/y:output:0*
T0*&
_output_shapes
:(X
truedivRealDivsub:z:0	mul_1:z:0*
T0*'
_output_shapes
:(J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
powPowtruediv:z:0pow/y:output:0*
T0*'
_output_shapes
:(L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ¿Y
mul_2Mulmul_2/x:output:0pow:z:0*
T0*'
_output_shapes
:(G
ExpExp	mul_2:z:0*
T0*'
_output_shapes
:(P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :t

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}(h
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"     (      h
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ¤
	depthwiseDepthwiseConv2dNativeExpandDims:output:0Exp:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(*
paddingSAME*
strides

  s
SqueezeSqueezedepthwise:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(*
squeeze_dims
c
IdentityIdentitySqueeze:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(e
NoOpNoOp^clip_by_value/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:ÿÿÿÿÿÿÿÿÿ}(: 2<
clip_by_value/ReadVariableOpclip_by_value/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}(
 
_user_specified_nameinputs
Ï®
â	
 __inference__wrapped_model_13462
input_1T
Baudio_classifier_leaf_tfbanks_complex_conv_readvariableop_resource:(g
Maudio_classifier_leaf_learnable_pooling_clip_by_value_readvariableop_resource:(H
:audio_classifier_leaf_pcen_minimum_readvariableop_resource:(H
:audio_classifier_leaf_pcen_maximum_readvariableop_resource:(R
Daudio_classifier_leaf_pcen_ema_clip_by_value_readvariableop_resource:(F
8audio_classifier_leaf_pcen_add_1_readvariableop_resource:(G
5audio_classifier_dense_matmul_readvariableop_resource:D
6audio_classifier_dense_biasadd_readvariableop_resource:
identity¢-audio_classifier/dense/BiasAdd/ReadVariableOp¢,audio_classifier/dense/MatMul/ReadVariableOp¢;audio_classifier/leaf/PCEN/EMA/clip_by_value/ReadVariableOp¢1audio_classifier/leaf/PCEN/Maximum/ReadVariableOp¢1audio_classifier/leaf/PCEN/Minimum/ReadVariableOp¢)audio_classifier/leaf/PCEN/ReadVariableOp¢/audio_classifier/leaf/PCEN/add_1/ReadVariableOp¢Daudio_classifier/leaf/learnable_pooling/clip_by_value/ReadVariableOp¢9audio_classifier/leaf/tfbanks_complex_conv/ReadVariableOp¢;audio_classifier/leaf/tfbanks_complex_conv/ReadVariableOp_1~
)audio_classifier/leaf/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
+audio_classifier/leaf/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            
+audio_classifier/leaf/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ×
#audio_classifier/leaf/strided_sliceStridedSliceinput_12audio_classifier/leaf/strided_slice/stack:output:04audio_classifier/leaf/strided_slice/stack_1:output:04audio_classifier/leaf/strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*

begin_mask*
end_mask*
new_axis_mask¼
9audio_classifier/leaf/tfbanks_complex_conv/ReadVariableOpReadVariableOpBaudio_classifier_leaf_tfbanks_complex_conv_readvariableop_resource*
_output_shapes

:(*
dtype0
>audio_classifier/leaf/tfbanks_complex_conv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
@audio_classifier/leaf/tfbanks_complex_conv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
@audio_classifier/leaf/tfbanks_complex_conv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
8audio_classifier/leaf/tfbanks_complex_conv/strided_sliceStridedSliceAaudio_classifier/leaf/tfbanks_complex_conv/ReadVariableOp:value:0Gaudio_classifier/leaf/tfbanks_complex_conv/strided_slice/stack:output:0Iaudio_classifier/leaf/tfbanks_complex_conv/strided_slice/stack_1:output:0Iaudio_classifier/leaf/tfbanks_complex_conv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:(*

begin_mask*
end_mask*
shrink_axis_mask
Baudio_classifier/leaf/tfbanks_complex_conv/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÛI@
@audio_classifier/leaf/tfbanks_complex_conv/clip_by_value/MinimumMinimumAaudio_classifier/leaf/tfbanks_complex_conv/strided_slice:output:0Kaudio_classifier/leaf/tfbanks_complex_conv/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:(
:audio_classifier/leaf/tfbanks_complex_conv/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ó
8audio_classifier/leaf/tfbanks_complex_conv/clip_by_valueMaximumDaudio_classifier/leaf/tfbanks_complex_conv/clip_by_value/Minimum:z:0Caudio_classifier/leaf/tfbanks_complex_conv/clip_by_value/y:output:0*
T0*
_output_shapes
:(¾
;audio_classifier/leaf/tfbanks_complex_conv/ReadVariableOp_1ReadVariableOpBaudio_classifier_leaf_tfbanks_complex_conv_readvariableop_resource*
_output_shapes

:(*
dtype0
@audio_classifier/leaf/tfbanks_complex_conv/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       
Baudio_classifier/leaf/tfbanks_complex_conv/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
Baudio_classifier/leaf/tfbanks_complex_conv/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      à
:audio_classifier/leaf/tfbanks_complex_conv/strided_slice_1StridedSliceCaudio_classifier/leaf/tfbanks_complex_conv/ReadVariableOp_1:value:0Iaudio_classifier/leaf/tfbanks_complex_conv/strided_slice_1/stack:output:0Kaudio_classifier/leaf/tfbanks_complex_conv/strided_slice_1/stack_1:output:0Kaudio_classifier/leaf/tfbanks_complex_conv/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:(*

begin_mask*
end_mask*
shrink_axis_mask
Daudio_classifier/leaf/tfbanks_complex_conv/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *IC
Baudio_classifier/leaf/tfbanks_complex_conv/clip_by_value_1/MinimumMinimumCaudio_classifier/leaf/tfbanks_complex_conv/strided_slice_1:output:0Maudio_classifier/leaf/tfbanks_complex_conv/clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:(
<audio_classifier/leaf/tfbanks_complex_conv/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *Tã¿?ù
:audio_classifier/leaf/tfbanks_complex_conv/clip_by_value_1MaximumFaudio_classifier/leaf/tfbanks_complex_conv/clip_by_value_1/Minimum:z:0Eaudio_classifier/leaf/tfbanks_complex_conv/clip_by_value_1/y:output:0*
T0*
_output_shapes
:(ô
0audio_classifier/leaf/tfbanks_complex_conv/stackPack<audio_classifier/leaf/tfbanks_complex_conv/clip_by_value:z:0>audio_classifier/leaf/tfbanks_complex_conv/clip_by_value_1:z:0*
N*
T0*
_output_shapes

:(*

axis{
6audio_classifier/leaf/tfbanks_complex_conv/range/startConst*
_output_shapes
: *
dtype0*
valueB
 *  HÃ{
6audio_classifier/leaf/tfbanks_complex_conv/range/limitConst*
_output_shapes
: *
dtype0*
valueB
 *  IC{
6audio_classifier/leaf/tfbanks_complex_conv/range/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¥
0audio_classifier/leaf/tfbanks_complex_conv/rangeRange?audio_classifier/leaf/tfbanks_complex_conv/range/start:output:0?audio_classifier/leaf/tfbanks_complex_conv/range/limit:output:0?audio_classifier/leaf/tfbanks_complex_conv/range/delta:output:0*

Tidx0*
_output_shapes	
:
@audio_classifier/leaf/tfbanks_complex_conv/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        
Baudio_classifier/leaf/tfbanks_complex_conv/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
Baudio_classifier/leaf/tfbanks_complex_conv/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
:audio_classifier/leaf/tfbanks_complex_conv/strided_slice_2StridedSlice9audio_classifier/leaf/tfbanks_complex_conv/stack:output:0Iaudio_classifier/leaf/tfbanks_complex_conv/strided_slice_2/stack:output:0Kaudio_classifier/leaf/tfbanks_complex_conv/strided_slice_2/stack_1:output:0Kaudio_classifier/leaf/tfbanks_complex_conv/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:(*

begin_mask*
end_mask*
shrink_axis_mask
@audio_classifier/leaf/tfbanks_complex_conv/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       
Baudio_classifier/leaf/tfbanks_complex_conv/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
Baudio_classifier/leaf/tfbanks_complex_conv/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ö
:audio_classifier/leaf/tfbanks_complex_conv/strided_slice_3StridedSlice9audio_classifier/leaf/tfbanks_complex_conv/stack:output:0Iaudio_classifier/leaf/tfbanks_complex_conv/strided_slice_3/stack:output:0Kaudio_classifier/leaf/tfbanks_complex_conv/strided_slice_3/stack_1:output:0Kaudio_classifier/leaf/tfbanks_complex_conv/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:(*

begin_mask*
end_mask*
shrink_axis_maskv
1audio_classifier/leaf/tfbanks_complex_conv/Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÛÉ@
/audio_classifier/leaf/tfbanks_complex_conv/SqrtSqrt:audio_classifier/leaf/tfbanks_complex_conv/Sqrt/x:output:0*
T0*
_output_shapes
: Ô
.audio_classifier/leaf/tfbanks_complex_conv/mulMul3audio_classifier/leaf/tfbanks_complex_conv/Sqrt:y:0Caudio_classifier/leaf/tfbanks_complex_conv/strided_slice_3:output:0*
T0*
_output_shapes
:(y
4audio_classifier/leaf/tfbanks_complex_conv/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Õ
2audio_classifier/leaf/tfbanks_complex_conv/truedivRealDiv=audio_classifier/leaf/tfbanks_complex_conv/truediv/x:output:02audio_classifier/leaf/tfbanks_complex_conv/mul:z:0*
T0*
_output_shapes
:(u
0audio_classifier/leaf/tfbanks_complex_conv/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @Ú
.audio_classifier/leaf/tfbanks_complex_conv/powPowCaudio_classifier/leaf/tfbanks_complex_conv/strided_slice_3:output:09audio_classifier/leaf/tfbanks_complex_conv/pow/y:output:0*
T0*
_output_shapes
:(w
2audio_classifier/leaf/tfbanks_complex_conv/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @Í
0audio_classifier/leaf/tfbanks_complex_conv/mul_1Mul;audio_classifier/leaf/tfbanks_complex_conv/mul_1/x:output:02audio_classifier/leaf/tfbanks_complex_conv/pow:z:0*
T0*
_output_shapes
:({
6audio_classifier/leaf/tfbanks_complex_conv/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Û
4audio_classifier/leaf/tfbanks_complex_conv/truediv_1RealDiv?audio_classifier/leaf/tfbanks_complex_conv/truediv_1/x:output:04audio_classifier/leaf/tfbanks_complex_conv/mul_1:z:0*
T0*
_output_shapes
:(w
2audio_classifier/leaf/tfbanks_complex_conv/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @Õ
0audio_classifier/leaf/tfbanks_complex_conv/pow_1Pow9audio_classifier/leaf/tfbanks_complex_conv/range:output:0;audio_classifier/leaf/tfbanks_complex_conv/pow_1/y:output:0*
T0*
_output_shapes	
:
.audio_classifier/leaf/tfbanks_complex_conv/NegNeg4audio_classifier/leaf/tfbanks_complex_conv/pow_1:z:0*
T0*
_output_shapes	
:
Baudio_classifier/leaf/tfbanks_complex_conv/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"(      ÷
<audio_classifier/leaf/tfbanks_complex_conv/Tensordot/ReshapeReshape8audio_classifier/leaf/tfbanks_complex_conv/truediv_1:z:0Kaudio_classifier/leaf/tfbanks_complex_conv/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:(
Daudio_classifier/leaf/tfbanks_complex_conv/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"     ö
>audio_classifier/leaf/tfbanks_complex_conv/Tensordot/Reshape_1Reshape2audio_classifier/leaf/tfbanks_complex_conv/Neg:y:0Maudio_classifier/leaf/tfbanks_complex_conv/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes
:	ÿ
;audio_classifier/leaf/tfbanks_complex_conv/Tensordot/MatMulMatMulEaudio_classifier/leaf/tfbanks_complex_conv/Tensordot/Reshape:output:0Gaudio_classifier/leaf/tfbanks_complex_conv/Tensordot/Reshape_1:output:0*
T0*
_output_shapes
:	(¦
.audio_classifier/leaf/tfbanks_complex_conv/ExpExpEaudio_classifier/leaf/tfbanks_complex_conv/Tensordot/MatMul:product:0*
T0*
_output_shapes
:	(°
/audio_classifier/leaf/tfbanks_complex_conv/CastCastCaudio_classifier/leaf/tfbanks_complex_conv/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
:(©
1audio_classifier/leaf/tfbanks_complex_conv/Cast_1Cast9audio_classifier/leaf/tfbanks_complex_conv/range:output:0*

DstT0*

SrcT0*
_output_shapes	
:
Daudio_classifier/leaf/tfbanks_complex_conv/Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"(      ö
>audio_classifier/leaf/tfbanks_complex_conv/Tensordot_1/ReshapeReshape3audio_classifier/leaf/tfbanks_complex_conv/Cast:y:0Maudio_classifier/leaf/tfbanks_complex_conv/Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes

:(
Faudio_classifier/leaf/tfbanks_complex_conv/Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"     ý
@audio_classifier/leaf/tfbanks_complex_conv/Tensordot_1/Reshape_1Reshape5audio_classifier/leaf/tfbanks_complex_conv/Cast_1:y:0Oaudio_classifier/leaf/tfbanks_complex_conv/Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes
:	
=audio_classifier/leaf/tfbanks_complex_conv/Tensordot_1/MatMulMatMulGaudio_classifier/leaf/tfbanks_complex_conv/Tensordot_1/Reshape:output:0Iaudio_classifier/leaf/tfbanks_complex_conv/Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes
:	({
2audio_classifier/leaf/tfbanks_complex_conv/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB J      ?ç
0audio_classifier/leaf/tfbanks_complex_conv/mul_2Mul;audio_classifier/leaf/tfbanks_complex_conv/mul_2/x:output:0Gaudio_classifier/leaf/tfbanks_complex_conv/Tensordot_1/MatMul:product:0*
T0*
_output_shapes
:	(
0audio_classifier/leaf/tfbanks_complex_conv/Exp_1Exp4audio_classifier/leaf/tfbanks_complex_conv/mul_2:z:0*
T0*
_output_shapes
:	(¥
1audio_classifier/leaf/tfbanks_complex_conv/Cast_2Cast6audio_classifier/leaf/tfbanks_complex_conv/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
:(
@audio_classifier/leaf/tfbanks_complex_conv/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        
Baudio_classifier/leaf/tfbanks_complex_conv/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
Baudio_classifier/leaf/tfbanks_complex_conv/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ó
:audio_classifier/leaf/tfbanks_complex_conv/strided_slice_4StridedSlice5audio_classifier/leaf/tfbanks_complex_conv/Cast_2:y:0Iaudio_classifier/leaf/tfbanks_complex_conv/strided_slice_4/stack:output:0Kaudio_classifier/leaf/tfbanks_complex_conv/strided_slice_4/stack_1:output:0Kaudio_classifier/leaf/tfbanks_complex_conv/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:(*

begin_mask*
end_mask*
new_axis_mask¦
1audio_classifier/leaf/tfbanks_complex_conv/Cast_3Cast2audio_classifier/leaf/tfbanks_complex_conv/Exp:y:0*

DstT0*

SrcT0*
_output_shapes
:	(Ü
0audio_classifier/leaf/tfbanks_complex_conv/mul_3MulCaudio_classifier/leaf/tfbanks_complex_conv/strided_slice_4:output:04audio_classifier/leaf/tfbanks_complex_conv/Exp_1:y:0*
T0*
_output_shapes
:	(Î
0audio_classifier/leaf/tfbanks_complex_conv/mul_4Mul4audio_classifier/leaf/tfbanks_complex_conv/mul_3:z:05audio_classifier/leaf/tfbanks_complex_conv/Cast_3:y:0*
T0*
_output_shapes
:	(
/audio_classifier/leaf/tfbanks_complex_conv/RealReal4audio_classifier/leaf/tfbanks_complex_conv/mul_4:z:0*
_output_shapes
:	(
/audio_classifier/leaf/tfbanks_complex_conv/ImagImag4audio_classifier/leaf/tfbanks_complex_conv/mul_4:z:0*
_output_shapes
:	(ñ
2audio_classifier/leaf/tfbanks_complex_conv/stack_1Pack8audio_classifier/leaf/tfbanks_complex_conv/Real:output:08audio_classifier/leaf/tfbanks_complex_conv/Imag:output:0*
N*
T0*#
_output_shapes
:(*

axis
8audio_classifier/leaf/tfbanks_complex_conv/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"P     ç
2audio_classifier/leaf/tfbanks_complex_conv/ReshapeReshape;audio_classifier/leaf/tfbanks_complex_conv/stack_1:output:0Aaudio_classifier/leaf/tfbanks_complex_conv/Reshape/shape:output:0*
T0*
_output_shapes
:	P
9audio_classifier/leaf/tfbanks_complex_conv/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       ì
4audio_classifier/leaf/tfbanks_complex_conv/transpose	Transpose;audio_classifier/leaf/tfbanks_complex_conv/Reshape:output:0Baudio_classifier/leaf/tfbanks_complex_conv/transpose/perm:output:0*
T0*
_output_shapes
:	P{
9audio_classifier/leaf/tfbanks_complex_conv/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :ï
5audio_classifier/leaf/tfbanks_complex_conv/ExpandDims
ExpandDims8audio_classifier/leaf/tfbanks_complex_conv/transpose:y:0Baudio_classifier/leaf/tfbanks_complex_conv/ExpandDims/dim:output:0*
T0*#
_output_shapes
:P
@audio_classifier/leaf/tfbanks_complex_conv/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿþ
<audio_classifier/leaf/tfbanks_complex_conv/conv1d/ExpandDims
ExpandDims,audio_classifier/leaf/strided_slice:output:0Iaudio_classifier/leaf/tfbanks_complex_conv/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
Baudio_classifier/leaf/tfbanks_complex_conv/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
>audio_classifier/leaf/tfbanks_complex_conv/conv1d/ExpandDims_1
ExpandDims>audio_classifier/leaf/tfbanks_complex_conv/ExpandDims:output:0Kaudio_classifier/leaf/tfbanks_complex_conv/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:P®
1audio_classifier/leaf/tfbanks_complex_conv/conv1dConv2DEaudio_classifier/leaf/tfbanks_complex_conv/conv1d/ExpandDims:output:0Gaudio_classifier/leaf/tfbanks_complex_conv/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}P*
paddingSAME*
strides
×
9audio_classifier/leaf/tfbanks_complex_conv/conv1d/SqueezeSqueeze:audio_classifier/leaf/tfbanks_complex_conv/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}P*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
4audio_classifier/leaf/squared_modulus/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ö
/audio_classifier/leaf/squared_modulus/transpose	TransposeBaudio_classifier/leaf/tfbanks_complex_conv/conv1d/Squeeze:output:0=audio_classifier/leaf/squared_modulus/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}p
+audio_classifier/leaf/squared_modulus/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @Ò
)audio_classifier/leaf/squared_modulus/powPow3audio_classifier/leaf/squared_modulus/transpose:y:04audio_classifier/leaf/squared_modulus/pow/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}
Faudio_classifier/leaf/squared_modulus/average_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :
Baudio_classifier/leaf/squared_modulus/average_pooling1d/ExpandDims
ExpandDims-audio_classifier/leaf/squared_modulus/pow:z:0Oaudio_classifier/leaf/squared_modulus/average_pooling1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}
?audio_classifier/leaf/squared_modulus/average_pooling1d/AvgPoolAvgPoolKaudio_classifier/leaf/squared_modulus/average_pooling1d/ExpandDims:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}*
ksize
*
paddingVALID*
strides
â
?audio_classifier/leaf/squared_modulus/average_pooling1d/SqueezeSqueezeHaudio_classifier/leaf/squared_modulus/average_pooling1d/AvgPool:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}*
squeeze_dims
p
+audio_classifier/leaf/squared_modulus/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @ç
)audio_classifier/leaf/squared_modulus/mulMul4audio_classifier/leaf/squared_modulus/mul/x:output:0Haudio_classifier/leaf/squared_modulus/average_pooling1d/Squeeze:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}
6audio_classifier/leaf/squared_modulus/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          å
1audio_classifier/leaf/squared_modulus/transpose_1	Transpose-audio_classifier/leaf/squared_modulus/mul:z:0?audio_classifier/leaf/squared_modulus/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}(Ú
Daudio_classifier/leaf/learnable_pooling/clip_by_value/ReadVariableOpReadVariableOpMaudio_classifier_leaf_learnable_pooling_clip_by_value_readvariableop_resource*&
_output_shapes
:(*
dtype0
?audio_classifier/leaf/learnable_pooling/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
=audio_classifier/leaf/learnable_pooling/clip_by_value/MinimumMinimumLaudio_classifier/leaf/learnable_pooling/clip_by_value/ReadVariableOp:value:0Haudio_classifier/leaf/learnable_pooling/clip_by_value/Minimum/y:output:0*
T0*&
_output_shapes
:(|
7audio_classifier/leaf/learnable_pooling/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *rn£;ö
5audio_classifier/leaf/learnable_pooling/clip_by_valueMaximumAaudio_classifier/leaf/learnable_pooling/clip_by_value/Minimum:z:0@audio_classifier/leaf/learnable_pooling/clip_by_value/y:output:0*
T0*&
_output_shapes
:(x
3audio_classifier/leaf/learnable_pooling/range/startConst*
_output_shapes
: *
dtype0*
valueB
 *    x
3audio_classifier/leaf/learnable_pooling/range/limitConst*
_output_shapes
: *
dtype0*
valueB
 * ÈCx
3audio_classifier/leaf/learnable_pooling/range/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
-audio_classifier/leaf/learnable_pooling/rangeRange<audio_classifier/leaf/learnable_pooling/range/start:output:0<audio_classifier/leaf/learnable_pooling/range/limit:output:0<audio_classifier/leaf/learnable_pooling/range/delta:output:0*

Tidx0*
_output_shapes	
:
5audio_classifier/leaf/learnable_pooling/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"           ä
/audio_classifier/leaf/learnable_pooling/ReshapeReshape6audio_classifier/leaf/learnable_pooling/range:output:0>audio_classifier/leaf/learnable_pooling/Reshape/shape:output:0*
T0*'
_output_shapes
:r
-audio_classifier/leaf/learnable_pooling/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HCÖ
+audio_classifier/leaf/learnable_pooling/subSub8audio_classifier/leaf/learnable_pooling/Reshape:output:06audio_classifier/leaf/learnable_pooling/sub/y:output:0*
T0*'
_output_shapes
:r
-audio_classifier/leaf/learnable_pooling/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ö
+audio_classifier/leaf/learnable_pooling/mulMul9audio_classifier/leaf/learnable_pooling/clip_by_value:z:06audio_classifier/leaf/learnable_pooling/mul/y:output:0*
T0*&
_output_shapes
:(t
/audio_classifier/leaf/learnable_pooling/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ÈCÐ
-audio_classifier/leaf/learnable_pooling/mul_1Mul/audio_classifier/leaf/learnable_pooling/mul:z:08audio_classifier/leaf/learnable_pooling/mul_1/y:output:0*
T0*&
_output_shapes
:(Ð
/audio_classifier/leaf/learnable_pooling/truedivRealDiv/audio_classifier/leaf/learnable_pooling/sub:z:01audio_classifier/leaf/learnable_pooling/mul_1:z:0*
T0*'
_output_shapes
:(r
-audio_classifier/leaf/learnable_pooling/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @Ñ
+audio_classifier/leaf/learnable_pooling/powPow3audio_classifier/leaf/learnable_pooling/truediv:z:06audio_classifier/leaf/learnable_pooling/pow/y:output:0*
T0*'
_output_shapes
:(t
/audio_classifier/leaf/learnable_pooling/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ¿Ñ
-audio_classifier/leaf/learnable_pooling/mul_2Mul8audio_classifier/leaf/learnable_pooling/mul_2/x:output:0/audio_classifier/leaf/learnable_pooling/pow:z:0*
T0*'
_output_shapes
:(
+audio_classifier/leaf/learnable_pooling/ExpExp1audio_classifier/leaf/learnable_pooling/mul_2:z:0*
T0*'
_output_shapes
:(x
6audio_classifier/leaf/learnable_pooling/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :ó
2audio_classifier/leaf/learnable_pooling/ExpandDims
ExpandDims5audio_classifier/leaf/squared_modulus/transpose_1:y:0?audio_classifier/leaf/learnable_pooling/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}(
7audio_classifier/leaf/learnable_pooling/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"     (      
?audio_classifier/leaf/learnable_pooling/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
1audio_classifier/leaf/learnable_pooling/depthwiseDepthwiseConv2dNative;audio_classifier/leaf/learnable_pooling/ExpandDims:output:0/audio_classifier/leaf/learnable_pooling/Exp:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(*
paddingSAME*
strides

  Ã
/audio_classifier/leaf/learnable_pooling/SqueezeSqueeze:audio_classifier/leaf/learnable_pooling/depthwise:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(*
squeeze_dims
d
audio_classifier/leaf/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7Â
audio_classifier/leaf/MaximumMaximum8audio_classifier/leaf/learnable_pooling/Squeeze:output:0(audio_classifier/leaf/Maximum/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(¨
1audio_classifier/leaf/PCEN/Minimum/ReadVariableOpReadVariableOp:audio_classifier_leaf_pcen_minimum_readvariableop_resource*
_output_shapes
:(*
dtype0i
$audio_classifier/leaf/PCEN/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¼
"audio_classifier/leaf/PCEN/MinimumMinimum9audio_classifier/leaf/PCEN/Minimum/ReadVariableOp:value:0-audio_classifier/leaf/PCEN/Minimum/y:output:0*
T0*
_output_shapes
:(¨
1audio_classifier/leaf/PCEN/Maximum/ReadVariableOpReadVariableOp:audio_classifier_leaf_pcen_maximum_readvariableop_resource*
_output_shapes
:(*
dtype0i
$audio_classifier/leaf/PCEN/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¼
"audio_classifier/leaf/PCEN/MaximumMaximum9audio_classifier/leaf/PCEN/Maximum/ReadVariableOp:value:0-audio_classifier/leaf/PCEN/Maximum/y:output:0*
T0*
_output_shapes
:(m
+audio_classifier/leaf/PCEN/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B : j
(audio_classifier/leaf/PCEN/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :
#audio_classifier/leaf/PCEN/GatherV2GatherV2!audio_classifier/leaf/Maximum:z:04audio_classifier/leaf/PCEN/GatherV2/indices:output:01audio_classifier/leaf/PCEN/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(¼
;audio_classifier/leaf/PCEN/EMA/clip_by_value/ReadVariableOpReadVariableOpDaudio_classifier_leaf_pcen_ema_clip_by_value_readvariableop_resource*
_output_shapes
:(*
dtype0{
6audio_classifier/leaf/PCEN/EMA/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ê
4audio_classifier/leaf/PCEN/EMA/clip_by_value/MinimumMinimumCaudio_classifier/leaf/PCEN/EMA/clip_by_value/ReadVariableOp:value:0?audio_classifier/leaf/PCEN/EMA/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:(s
.audio_classifier/leaf/PCEN/EMA/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    Ï
,audio_classifier/leaf/PCEN/EMA/clip_by_valueMaximum8audio_classifier/leaf/PCEN/EMA/clip_by_value/Minimum:z:07audio_classifier/leaf/PCEN/EMA/clip_by_value/y:output:0*
T0*
_output_shapes
:(
-audio_classifier/leaf/PCEN/EMA/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Æ
(audio_classifier/leaf/PCEN/EMA/transpose	Transpose!audio_classifier/leaf/Maximum:z:06audio_classifier/leaf/PCEN/EMA/transpose/perm:output:0*
T0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿ(
?audio_classifier/leaf/PCEN/EMA/scan/TensorArrayV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   
>audio_classifier/leaf/PCEN/EMA/scan/TensorArrayV2/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :d«
1audio_classifier/leaf/PCEN/EMA/scan/TensorArrayV2TensorListReserveHaudio_classifier/leaf/PCEN/EMA/scan/TensorArrayV2/element_shape:output:0Gaudio_classifier/leaf/PCEN/EMA/scan/TensorArrayV2/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒª
Yaudio_classifier/leaf/PCEN/EMA/scan/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   Ç
Kaudio_classifier/leaf/PCEN/EMA/scan/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor,audio_classifier/leaf/PCEN/EMA/transpose:y:0baudio_classifier/leaf/PCEN/EMA/scan/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Aaudio_classifier/leaf/PCEN/EMA/scan/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   
@audio_classifier/leaf/PCEN/EMA/scan/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :d±
3audio_classifier/leaf/PCEN/EMA/scan/TensorArrayV2_1TensorListReserveJaudio_classifier/leaf/PCEN/EMA/scan/TensorArrayV2_1/element_shape:output:0Iaudio_classifier/leaf/PCEN/EMA/scan/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒk
)audio_classifier/leaf/PCEN/EMA/scan/ConstConst*
_output_shapes
: *
dtype0*
value	B : ~
<audio_classifier/leaf/PCEN/EMA/scan/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
value	B :dx
6audio_classifier/leaf/PCEN/EMA/scan/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ö
)audio_classifier/leaf/PCEN/EMA/scan/whileStatelessWhile?audio_classifier/leaf/PCEN/EMA/scan/while/loop_counter:output:0Eaudio_classifier/leaf/PCEN/EMA/scan/while/maximum_iterations:output:02audio_classifier/leaf/PCEN/EMA/scan/Const:output:0,audio_classifier/leaf/PCEN/GatherV2:output:0<audio_classifier/leaf/PCEN/EMA/scan/TensorArrayV2_1:handle:0[audio_classifier/leaf/PCEN/EMA/scan/TensorArrayUnstack/TensorListFromTensor:output_handle:00audio_classifier/leaf/PCEN/EMA/clip_by_value:z:0*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*7
_output_shapes%
#: : : :ÿÿÿÿÿÿÿÿÿ(: : :(* 
_read_only_resource_inputs
 *
_stateful_parallelism( *@
body8R6
4audio_classifier_leaf_PCEN_EMA_scan_while_body_13384*@
cond8R6
4audio_classifier_leaf_PCEN_EMA_scan_while_cond_13383*6
output_shapes%
#: : : :ÿÿÿÿÿÿÿÿÿ(: : :(¥
Taudio_classifier/leaf/PCEN/EMA/scan/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   Â
Faudio_classifier/leaf/PCEN/EMA/scan/TensorArrayV2Stack/TensorListStackTensorListStack2audio_classifier/leaf/PCEN/EMA/scan/while:output:4]audio_classifier/leaf/PCEN/EMA/scan/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿ(*
element_dtype0*
num_elementsd
/audio_classifier/leaf/PCEN/EMA/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ø
*audio_classifier/leaf/PCEN/EMA/transpose_1	TransposeOaudio_classifier/leaf/PCEN/EMA/scan/TensorArrayV2Stack/TensorListStack:tensor:08audio_classifier/leaf/PCEN/EMA/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(i
$audio_classifier/leaf/PCEN/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?©
"audio_classifier/leaf/PCEN/truedivRealDiv-audio_classifier/leaf/PCEN/truediv/x:output:0&audio_classifier/leaf/PCEN/Maximum:z:0*
T0*
_output_shapes
:(e
 audio_classifier/leaf/PCEN/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ì¼+¸
audio_classifier/leaf/PCEN/addAddV2)audio_classifier/leaf/PCEN/add/x:output:0.audio_classifier/leaf/PCEN/EMA/transpose_1:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(§
audio_classifier/leaf/PCEN/powPow"audio_classifier/leaf/PCEN/add:z:0&audio_classifier/leaf/PCEN/Minimum:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(¬
$audio_classifier/leaf/PCEN/truediv_1RealDiv!audio_classifier/leaf/Maximum:z:0"audio_classifier/leaf/PCEN/pow:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(¤
/audio_classifier/leaf/PCEN/add_1/ReadVariableOpReadVariableOp8audio_classifier_leaf_pcen_add_1_readvariableop_resource*
_output_shapes
:(*
dtype0Â
 audio_classifier/leaf/PCEN/add_1AddV2(audio_classifier/leaf/PCEN/truediv_1:z:07audio_classifier/leaf/PCEN/add_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(«
 audio_classifier/leaf/PCEN/pow_1Pow$audio_classifier/leaf/PCEN/add_1:z:0&audio_classifier/leaf/PCEN/truediv:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(
)audio_classifier/leaf/PCEN/ReadVariableOpReadVariableOp8audio_classifier_leaf_pcen_add_1_readvariableop_resource*
_output_shapes
:(*
dtype0§
 audio_classifier/leaf/PCEN/pow_2Pow1audio_classifier/leaf/PCEN/ReadVariableOp:value:0&audio_classifier/leaf/PCEN/truediv:z:0*
T0*
_output_shapes
:(§
audio_classifier/leaf/PCEN/subSub$audio_classifier/leaf/PCEN/pow_1:z:0$audio_classifier/leaf/PCEN/pow_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(j
audio_classifier/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ±
audio_classifier/ExpandDims
ExpandDims"audio_classifier/leaf/PCEN/sub:z:0(audio_classifier/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(
Faudio_classifier/sequential/global_max_pooling2d/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ä
4audio_classifier/sequential/global_max_pooling2d/MaxMax$audio_classifier/ExpandDims:output:0Oaudio_classifier/sequential/global_max_pooling2d/Max/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
)audio_classifier/sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Û
+audio_classifier/sequential/flatten/ReshapeReshape=audio_classifier/sequential/global_max_pooling2d/Max:output:02audio_classifier/sequential/flatten/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
,audio_classifier/dense/MatMul/ReadVariableOpReadVariableOp5audio_classifier_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Å
audio_classifier/dense/MatMulMatMul4audio_classifier/sequential/flatten/Reshape:output:04audio_classifier/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
-audio_classifier/dense/BiasAdd/ReadVariableOpReadVariableOp6audio_classifier_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0»
audio_classifier/dense/BiasAddBiasAdd'audio_classifier/dense/MatMul:product:05audio_classifier/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
IdentityIdentity'audio_classifier/dense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp.^audio_classifier/dense/BiasAdd/ReadVariableOp-^audio_classifier/dense/MatMul/ReadVariableOp<^audio_classifier/leaf/PCEN/EMA/clip_by_value/ReadVariableOp2^audio_classifier/leaf/PCEN/Maximum/ReadVariableOp2^audio_classifier/leaf/PCEN/Minimum/ReadVariableOp*^audio_classifier/leaf/PCEN/ReadVariableOp0^audio_classifier/leaf/PCEN/add_1/ReadVariableOpE^audio_classifier/leaf/learnable_pooling/clip_by_value/ReadVariableOp:^audio_classifier/leaf/tfbanks_complex_conv/ReadVariableOp<^audio_classifier/leaf/tfbanks_complex_conv/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ}: : : : : : : : 2^
-audio_classifier/dense/BiasAdd/ReadVariableOp-audio_classifier/dense/BiasAdd/ReadVariableOp2\
,audio_classifier/dense/MatMul/ReadVariableOp,audio_classifier/dense/MatMul/ReadVariableOp2z
;audio_classifier/leaf/PCEN/EMA/clip_by_value/ReadVariableOp;audio_classifier/leaf/PCEN/EMA/clip_by_value/ReadVariableOp2f
1audio_classifier/leaf/PCEN/Maximum/ReadVariableOp1audio_classifier/leaf/PCEN/Maximum/ReadVariableOp2f
1audio_classifier/leaf/PCEN/Minimum/ReadVariableOp1audio_classifier/leaf/PCEN/Minimum/ReadVariableOp2V
)audio_classifier/leaf/PCEN/ReadVariableOp)audio_classifier/leaf/PCEN/ReadVariableOp2b
/audio_classifier/leaf/PCEN/add_1/ReadVariableOp/audio_classifier/leaf/PCEN/add_1/ReadVariableOp2
Daudio_classifier/leaf/learnable_pooling/clip_by_value/ReadVariableOpDaudio_classifier/leaf/learnable_pooling/clip_by_value/ReadVariableOp2v
9audio_classifier/leaf/tfbanks_complex_conv/ReadVariableOp9audio_classifier/leaf/tfbanks_complex_conv/ReadVariableOp2z
;audio_classifier/leaf/tfbanks_complex_conv/ReadVariableOp_1;audio_classifier/leaf/tfbanks_complex_conv/ReadVariableOp_1:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
!
_user_specified_name	input_1
ñ

$__inference_leaf_layer_call_fn_14795

inputs
unknown:(#
	unknown_0:(
	unknown_1:(
	unknown_2:(
	unknown_3:(
	unknown_4:(
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_leaf_layer_call_and_return_conditional_losses_13748s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ}: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
 
_user_specified_nameinputs
6

?__inference_PCEN_layer_call_and_return_conditional_losses_15589

inputs-
minimum_readvariableop_resource:(-
maximum_readvariableop_resource:(7
)ema_clip_by_value_readvariableop_resource:(+
add_1_readvariableop_resource:(
identity¢ EMA/clip_by_value/ReadVariableOp¢Maximum/ReadVariableOp¢Minimum/ReadVariableOp¢ReadVariableOp¢add_1/ReadVariableOpr
Minimum/ReadVariableOpReadVariableOpminimum_readvariableop_resource*
_output_shapes
:(*
dtype0N
	Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?k
MinimumMinimumMinimum/ReadVariableOp:value:0Minimum/y:output:0*
T0*
_output_shapes
:(r
Maximum/ReadVariableOpReadVariableOpmaximum_readvariableop_resource*
_output_shapes
:(*
dtype0N
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?k
MaximumMaximumMaximum/ReadVariableOp:value:0Maximum/y:output:0*
T0*
_output_shapes
:(R
GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B : O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :¤
GatherV2GatherV2inputsGatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 EMA/clip_by_value/ReadVariableOpReadVariableOp)ema_clip_by_value_readvariableop_resource*
_output_shapes
:(*
dtype0`
EMA/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
EMA/clip_by_value/MinimumMinimum(EMA/clip_by_value/ReadVariableOp:value:0$EMA/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:(X
EMA/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
EMA/clip_by_valueMaximumEMA/clip_by_value/Minimum:z:0EMA/clip_by_value/y:output:0*
T0*
_output_shapes
:(g
EMA/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          u
EMA/transpose	TransposeinputsEMA/transpose/perm:output:0*
T0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿ(u
$EMA/scan/TensorArrayV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   e
#EMA/scan/TensorArrayV2/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :dÚ
EMA/scan/TensorArrayV2TensorListReserve-EMA/scan/TensorArrayV2/element_shape:output:0,EMA/scan/TensorArrayV2/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
>EMA/scan/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   ö
0EMA/scan/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorEMA/transpose:y:0GEMA/scan/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒw
&EMA/scan/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   g
%EMA/scan/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :dà
EMA/scan/TensorArrayV2_1TensorListReserve/EMA/scan/TensorArrayV2_1/element_shape:output:0.EMA/scan/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒP
EMA/scan/ConstConst*
_output_shapes
: *
dtype0*
value	B : c
!EMA/scan/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
value	B :d]
EMA/scan/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : è
EMA/scan/whileStatelessWhile$EMA/scan/while/loop_counter:output:0*EMA/scan/while/maximum_iterations:output:0EMA/scan/Const:output:0GatherV2:output:0!EMA/scan/TensorArrayV2_1:handle:0@EMA/scan/TensorArrayUnstack/TensorListFromTensor:output_handle:0EMA/clip_by_value:z:0*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*7
_output_shapes%
#: : : :ÿÿÿÿÿÿÿÿÿ(: : :(* 
_read_only_resource_inputs
 *
_stateful_parallelism( *%
bodyR
EMA_scan_while_body_15523*%
condR
EMA_scan_while_cond_15522*6
output_shapes%
#: : : :ÿÿÿÿÿÿÿÿÿ(: : :(
9EMA/scan/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   ñ
+EMA/scan/TensorArrayV2Stack/TensorListStackTensorListStackEMA/scan/while:output:4BEMA/scan/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿ(*
element_dtype0*
num_elementsdi
EMA/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          §
EMA/transpose_1	Transpose4EMA/scan/TensorArrayV2Stack/TensorListStack:tensor:0EMA/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(N
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?X
truedivRealDivtruediv/x:output:0Maximum:z:0*
T0*
_output_shapes
:(J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ì¼+g
addAddV2add/x:output:0EMA/transpose_1:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(V
powPowadd:z:0Minimum:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd([
	truediv_1RealDivinputspow:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(n
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:(*
dtype0q
add_1AddV2truediv_1:z:0add_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(Z
pow_1Pow	add_1:z:0truediv:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(h
ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:(*
dtype0V
pow_2PowReadVariableOp:value:0truediv:z:0*
T0*
_output_shapes
:(V
subSub	pow_1:z:0	pow_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(Z
IdentityIdentitysub:z:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(Ã
NoOpNoOp!^EMA/clip_by_value/ReadVariableOp^Maximum/ReadVariableOp^Minimum/ReadVariableOp^ReadVariableOp^add_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿd(: : : : 2D
 EMA/clip_by_value/ReadVariableOp EMA/clip_by_value/ReadVariableOp20
Maximum/ReadVariableOpMaximum/ReadVariableOp20
Minimum/ReadVariableOpMinimum/ReadVariableOp2 
ReadVariableOpReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(
 
_user_specified_nameinputs
Ä

K__inference_audio_classifier_layer_call_and_return_conditional_losses_14130

inputs

leaf_14108:($

leaf_14110:(

leaf_14112:(

leaf_14114:(

leaf_14116:(

leaf_14118:(
dense_14124:
dense_14126:
identity¢dense/StatefulPartitionedCall¢leaf/StatefulPartitionedCall
leaf/StatefulPartitionedCallStatefulPartitionedCallinputs
leaf_14108
leaf_14110
leaf_14112
leaf_14114
leaf_14116
leaf_14118*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_leaf_layer_call_and_return_conditional_losses_13845Y
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ

ExpandDims
ExpandDims%leaf/StatefulPartitionedCall:output:0ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(Ë
sequential/PartitionedCallPartitionedCallExpandDims:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_13993
dense/StatefulPartitionedCallStatefulPartitionedCall#sequential/PartitionedCall:output:0dense_14124dense_14126*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_14046u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense/StatefulPartitionedCall^leaf/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ}: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2<
leaf/StatefulPartitionedCallleaf/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
 
_user_specified_nameinputs
	
u
E__inference_sequential_layer_call_and_return_conditional_losses_14013
global_max_pooling2d_input
identityæ
$global_max_pooling2d/PartitionedCallPartitionedCallglobal_max_pooling2d_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_13955ß
flatten/PartitionedCallPartitionedCall-global_max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_13963h
IdentityIdentity flatten/PartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd(:k g
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(
4
_user_specified_nameglobal_max_pooling2d_input


#leaf_PCEN_EMA_scan_while_cond_14455B
>leaf_pcen_ema_scan_while_leaf_pcen_ema_scan_while_loop_counterH
Dleaf_pcen_ema_scan_while_leaf_pcen_ema_scan_while_maximum_iterations(
$leaf_pcen_ema_scan_while_placeholder*
&leaf_pcen_ema_scan_while_placeholder_1*
&leaf_pcen_ema_scan_while_placeholder_2Y
Uleaf_pcen_ema_scan_while_leaf_pcen_ema_scan_while_cond_14455___redundant_placeholder0Y
Uleaf_pcen_ema_scan_while_leaf_pcen_ema_scan_while_cond_14455___redundant_placeholder1%
!leaf_pcen_ema_scan_while_identity
a
leaf/PCEN/EMA/scan/while/Less/yConst*
_output_shapes
: *
dtype0*
value	B :d
leaf/PCEN/EMA/scan/while/LessLess$leaf_pcen_ema_scan_while_placeholder(leaf/PCEN/EMA/scan/while/Less/y:output:0*
T0*
_output_shapes
: Î
leaf/PCEN/EMA/scan/while/Less_1Less>leaf_pcen_ema_scan_while_leaf_pcen_ema_scan_while_loop_counterDleaf_pcen_ema_scan_while_leaf_pcen_ema_scan_while_maximum_iterations*
T0*
_output_shapes
: 
#leaf/PCEN/EMA/scan/while/LogicalAnd
LogicalAnd#leaf/PCEN/EMA/scan/while/Less_1:z:0!leaf/PCEN/EMA/scan/while/Less:z:0*
_output_shapes
: w
!leaf/PCEN/EMA/scan/while/IdentityIdentity'leaf/PCEN/EMA/scan/while/LogicalAnd:z:0*
T0
*
_output_shapes
: "O
!leaf_pcen_ema_scan_while_identity*leaf/PCEN/EMA/scan/while/Identity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : :ÿÿÿÿÿÿÿÿÿ(: ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:

_output_shapes
: :

_output_shapes
::

_output_shapes
:
×"

PCEN_EMA_scan_while_body_152088
4pcen_ema_scan_while_pcen_ema_scan_while_loop_counter>
:pcen_ema_scan_while_pcen_ema_scan_while_maximum_iterations#
pcen_ema_scan_while_placeholder%
!pcen_ema_scan_while_placeholder_1%
!pcen_ema_scan_while_placeholder_2s
opcen_ema_scan_while_tensorarrayv2read_tensorlistgetitem_pcen_ema_scan_tensorarrayunstack_tensorlistfromtensor_04
0pcen_ema_scan_while_mul_pcen_ema_clip_by_value_0 
pcen_ema_scan_while_identity"
pcen_ema_scan_while_identity_1"
pcen_ema_scan_while_identity_2"
pcen_ema_scan_while_identity_3"
pcen_ema_scan_while_identity_4q
mpcen_ema_scan_while_tensorarrayv2read_tensorlistgetitem_pcen_ema_scan_tensorarrayunstack_tensorlistfromtensor2
.pcen_ema_scan_while_mul_pcen_ema_clip_by_value
EPCEN/EMA/scan/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   ì
7PCEN/EMA/scan/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemopcen_ema_scan_while_tensorarrayv2read_tensorlistgetitem_pcen_ema_scan_tensorarrayunstack_tensorlistfromtensor_0pcen_ema_scan_while_placeholderNPCEN/EMA/scan/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
element_dtype0Â
PCEN/EMA/scan/while/mulMul0pcen_ema_scan_while_mul_pcen_ema_clip_by_value_0>PCEN/EMA/scan/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(^
PCEN/EMA/scan/while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
PCEN/EMA/scan/while/subSub"PCEN/EMA/scan/while/sub/x:output:00pcen_ema_scan_while_mul_pcen_ema_clip_by_value_0*
T0*
_output_shapes
:(
PCEN/EMA/scan/while/mul_1MulPCEN/EMA/scan/while/sub:z:0!pcen_ema_scan_while_placeholder_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
PCEN/EMA/scan/while/addAddV2PCEN/EMA/scan/while/mul:z:0PCEN/EMA/scan/while/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(î
8PCEN/EMA/scan/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!pcen_ema_scan_while_placeholder_2pcen_ema_scan_while_placeholderPCEN/EMA/scan/while/add:z:0*
_output_shapes
: *
element_dtype0:éèÒ]
PCEN/EMA/scan/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
PCEN/EMA/scan/while/add_1AddV2pcen_ema_scan_while_placeholder$PCEN/EMA/scan/while/add_1/y:output:0*
T0*
_output_shapes
: ]
PCEN/EMA/scan/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :
PCEN/EMA/scan/while/add_2AddV24pcen_ema_scan_while_pcen_ema_scan_while_loop_counter$PCEN/EMA/scan/while/add_2/y:output:0*
T0*
_output_shapes
: h
PCEN/EMA/scan/while/IdentityIdentityPCEN/EMA/scan/while/add_2:z:0*
T0*
_output_shapes
: 
PCEN/EMA/scan/while/Identity_1Identity:pcen_ema_scan_while_pcen_ema_scan_while_maximum_iterations*
T0*
_output_shapes
: j
PCEN/EMA/scan/while/Identity_2IdentityPCEN/EMA/scan/while/add_1:z:0*
T0*
_output_shapes
: y
PCEN/EMA/scan/while/Identity_3IdentityPCEN/EMA/scan/while/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
PCEN/EMA/scan/while/Identity_4IdentityHPCEN/EMA/scan/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "E
pcen_ema_scan_while_identity%PCEN/EMA/scan/while/Identity:output:0"I
pcen_ema_scan_while_identity_1'PCEN/EMA/scan/while/Identity_1:output:0"I
pcen_ema_scan_while_identity_2'PCEN/EMA/scan/while/Identity_2:output:0"I
pcen_ema_scan_while_identity_3'PCEN/EMA/scan/while/Identity_3:output:0"I
pcen_ema_scan_while_identity_4'PCEN/EMA/scan/while/Identity_4:output:0"b
.pcen_ema_scan_while_mul_pcen_ema_clip_by_value0pcen_ema_scan_while_mul_pcen_ema_clip_by_value_0"à
mpcen_ema_scan_while_tensorarrayv2read_tensorlistgetitem_pcen_ema_scan_tensorarrayunstack_tensorlistfromtensoropcen_ema_scan_while_tensorarrayv2read_tensorlistgetitem_pcen_ema_scan_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : :ÿÿÿÿÿÿÿÿÿ(: : :(: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:(
×"

PCEN_EMA_scan_while_body_149778
4pcen_ema_scan_while_pcen_ema_scan_while_loop_counter>
:pcen_ema_scan_while_pcen_ema_scan_while_maximum_iterations#
pcen_ema_scan_while_placeholder%
!pcen_ema_scan_while_placeholder_1%
!pcen_ema_scan_while_placeholder_2s
opcen_ema_scan_while_tensorarrayv2read_tensorlistgetitem_pcen_ema_scan_tensorarrayunstack_tensorlistfromtensor_04
0pcen_ema_scan_while_mul_pcen_ema_clip_by_value_0 
pcen_ema_scan_while_identity"
pcen_ema_scan_while_identity_1"
pcen_ema_scan_while_identity_2"
pcen_ema_scan_while_identity_3"
pcen_ema_scan_while_identity_4q
mpcen_ema_scan_while_tensorarrayv2read_tensorlistgetitem_pcen_ema_scan_tensorarrayunstack_tensorlistfromtensor2
.pcen_ema_scan_while_mul_pcen_ema_clip_by_value
EPCEN/EMA/scan/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   ì
7PCEN/EMA/scan/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemopcen_ema_scan_while_tensorarrayv2read_tensorlistgetitem_pcen_ema_scan_tensorarrayunstack_tensorlistfromtensor_0pcen_ema_scan_while_placeholderNPCEN/EMA/scan/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
element_dtype0Â
PCEN/EMA/scan/while/mulMul0pcen_ema_scan_while_mul_pcen_ema_clip_by_value_0>PCEN/EMA/scan/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(^
PCEN/EMA/scan/while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
PCEN/EMA/scan/while/subSub"PCEN/EMA/scan/while/sub/x:output:00pcen_ema_scan_while_mul_pcen_ema_clip_by_value_0*
T0*
_output_shapes
:(
PCEN/EMA/scan/while/mul_1MulPCEN/EMA/scan/while/sub:z:0!pcen_ema_scan_while_placeholder_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
PCEN/EMA/scan/while/addAddV2PCEN/EMA/scan/while/mul:z:0PCEN/EMA/scan/while/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(î
8PCEN/EMA/scan/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!pcen_ema_scan_while_placeholder_2pcen_ema_scan_while_placeholderPCEN/EMA/scan/while/add:z:0*
_output_shapes
: *
element_dtype0:éèÒ]
PCEN/EMA/scan/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
PCEN/EMA/scan/while/add_1AddV2pcen_ema_scan_while_placeholder$PCEN/EMA/scan/while/add_1/y:output:0*
T0*
_output_shapes
: ]
PCEN/EMA/scan/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :
PCEN/EMA/scan/while/add_2AddV24pcen_ema_scan_while_pcen_ema_scan_while_loop_counter$PCEN/EMA/scan/while/add_2/y:output:0*
T0*
_output_shapes
: h
PCEN/EMA/scan/while/IdentityIdentityPCEN/EMA/scan/while/add_2:z:0*
T0*
_output_shapes
: 
PCEN/EMA/scan/while/Identity_1Identity:pcen_ema_scan_while_pcen_ema_scan_while_maximum_iterations*
T0*
_output_shapes
: j
PCEN/EMA/scan/while/Identity_2IdentityPCEN/EMA/scan/while/add_1:z:0*
T0*
_output_shapes
: y
PCEN/EMA/scan/while/Identity_3IdentityPCEN/EMA/scan/while/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
PCEN/EMA/scan/while/Identity_4IdentityHPCEN/EMA/scan/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "E
pcen_ema_scan_while_identity%PCEN/EMA/scan/while/Identity:output:0"I
pcen_ema_scan_while_identity_1'PCEN/EMA/scan/while/Identity_1:output:0"I
pcen_ema_scan_while_identity_2'PCEN/EMA/scan/while/Identity_2:output:0"I
pcen_ema_scan_while_identity_3'PCEN/EMA/scan/while/Identity_3:output:0"I
pcen_ema_scan_while_identity_4'PCEN/EMA/scan/while/Identity_4:output:0"b
.pcen_ema_scan_while_mul_pcen_ema_clip_by_value0pcen_ema_scan_while_mul_pcen_ema_clip_by_value_0"à
mpcen_ema_scan_while_tensorarrayv2read_tensorlistgetitem_pcen_ema_scan_tensorarrayunstack_tensorlistfromtensoropcen_ema_scan_while_tensorarrayv2read_tensorlistgetitem_pcen_ema_scan_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : :ÿÿÿÿÿÿÿÿÿ(: : :(: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:(
Õ	
Ã
0__inference_audio_classifier_layer_call_fn_14270

inputs
unknown:(#
	unknown_0:(
	unknown_1:(
	unknown_2:(
	unknown_3:(
	unknown_4:(
	unknown_5:
	unknown_6:
identity¢StatefulPartitionedCall±
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_audio_classifier_layer_call_and_return_conditional_losses_14053o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ}: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
 
_user_specified_nameinputs

P
4__inference_global_max_pooling2d_layer_call_fn_15594

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_13940i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
u
E__inference_sequential_layer_call_and_return_conditional_losses_14007
global_max_pooling2d_input
identityæ
$global_max_pooling2d/PartitionedCallPartitionedCallglobal_max_pooling2d_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_13955ß
flatten/PartitionedCallPartitionedCall-global_max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_13963h
IdentityIdentity flatten/PartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd(:k g
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(
4
_user_specified_nameglobal_max_pooling2d_input
²
^
B__inference_flatten_layer_call_and_return_conditional_losses_15622

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã	
ñ
@__inference_dense_layer_call_and_return_conditional_losses_14046

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
î
Z
*__inference_sequential_layer_call_fn_14001
global_max_pooling2d_input
identityÇ
PartitionedCallPartitionedCallglobal_max_pooling2d_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_13993`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd(:k g
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(
4
_user_specified_nameglobal_max_pooling2d_input
­Ø
à
?__inference_leaf_layer_call_and_return_conditional_losses_15275

inputs>
,tfbanks_complex_conv_readvariableop_resource:(Q
7learnable_pooling_clip_by_value_readvariableop_resource:(2
$pcen_minimum_readvariableop_resource:(2
$pcen_maximum_readvariableop_resource:(<
.pcen_ema_clip_by_value_readvariableop_resource:(0
"pcen_add_1_readvariableop_resource:(

identity_1¢%PCEN/EMA/clip_by_value/ReadVariableOp¢PCEN/Maximum/ReadVariableOp¢PCEN/Minimum/ReadVariableOp¢PCEN/ReadVariableOp¢PCEN/add_1/ReadVariableOp¢.learnable_pooling/clip_by_value/ReadVariableOp¢#tfbanks_complex_conv/ReadVariableOp¢%tfbanks_complex_conv/ReadVariableOp_1h
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         þ
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*

begin_mask*
end_mask*
new_axis_mask
#tfbanks_complex_conv/ReadVariableOpReadVariableOp,tfbanks_complex_conv_readvariableop_resource*
_output_shapes

:(*
dtype0y
(tfbanks_complex_conv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        {
*tfbanks_complex_conv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*tfbanks_complex_conv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      è
"tfbanks_complex_conv/strided_sliceStridedSlice+tfbanks_complex_conv/ReadVariableOp:value:01tfbanks_complex_conv/strided_slice/stack:output:03tfbanks_complex_conv/strided_slice/stack_1:output:03tfbanks_complex_conv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:(*

begin_mask*
end_mask*
shrink_axis_maskq
,tfbanks_complex_conv/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÛI@¾
*tfbanks_complex_conv/clip_by_value/MinimumMinimum+tfbanks_complex_conv/strided_slice:output:05tfbanks_complex_conv/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:(i
$tfbanks_complex_conv/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ±
"tfbanks_complex_conv/clip_by_valueMaximum.tfbanks_complex_conv/clip_by_value/Minimum:z:0-tfbanks_complex_conv/clip_by_value/y:output:0*
T0*
_output_shapes
:(
%tfbanks_complex_conv/ReadVariableOp_1ReadVariableOp,tfbanks_complex_conv_readvariableop_resource*
_output_shapes

:(*
dtype0{
*tfbanks_complex_conv/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,tfbanks_complex_conv/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,tfbanks_complex_conv/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ò
$tfbanks_complex_conv/strided_slice_1StridedSlice-tfbanks_complex_conv/ReadVariableOp_1:value:03tfbanks_complex_conv/strided_slice_1/stack:output:05tfbanks_complex_conv/strided_slice_1/stack_1:output:05tfbanks_complex_conv/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:(*

begin_mask*
end_mask*
shrink_axis_masks
.tfbanks_complex_conv/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ICÄ
,tfbanks_complex_conv/clip_by_value_1/MinimumMinimum-tfbanks_complex_conv/strided_slice_1:output:07tfbanks_complex_conv/clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:(k
&tfbanks_complex_conv/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *Tã¿?·
$tfbanks_complex_conv/clip_by_value_1Maximum0tfbanks_complex_conv/clip_by_value_1/Minimum:z:0/tfbanks_complex_conv/clip_by_value_1/y:output:0*
T0*
_output_shapes
:(²
tfbanks_complex_conv/stackPack&tfbanks_complex_conv/clip_by_value:z:0(tfbanks_complex_conv/clip_by_value_1:z:0*
N*
T0*
_output_shapes

:(*

axise
 tfbanks_complex_conv/range/startConst*
_output_shapes
: *
dtype0*
valueB
 *  HÃe
 tfbanks_complex_conv/range/limitConst*
_output_shapes
: *
dtype0*
valueB
 *  ICe
 tfbanks_complex_conv/range/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Í
tfbanks_complex_conv/rangeRange)tfbanks_complex_conv/range/start:output:0)tfbanks_complex_conv/range/limit:output:0)tfbanks_complex_conv/range/delta:output:0*

Tidx0*
_output_shapes	
:{
*tfbanks_complex_conv/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        }
,tfbanks_complex_conv/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,tfbanks_complex_conv/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      è
$tfbanks_complex_conv/strided_slice_2StridedSlice#tfbanks_complex_conv/stack:output:03tfbanks_complex_conv/strided_slice_2/stack:output:05tfbanks_complex_conv/strided_slice_2/stack_1:output:05tfbanks_complex_conv/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:(*

begin_mask*
end_mask*
shrink_axis_mask{
*tfbanks_complex_conv/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,tfbanks_complex_conv/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,tfbanks_complex_conv/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      è
$tfbanks_complex_conv/strided_slice_3StridedSlice#tfbanks_complex_conv/stack:output:03tfbanks_complex_conv/strided_slice_3/stack:output:05tfbanks_complex_conv/strided_slice_3/stack_1:output:05tfbanks_complex_conv/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:(*

begin_mask*
end_mask*
shrink_axis_mask`
tfbanks_complex_conv/Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÛÉ@h
tfbanks_complex_conv/SqrtSqrt$tfbanks_complex_conv/Sqrt/x:output:0*
T0*
_output_shapes
: 
tfbanks_complex_conv/mulMultfbanks_complex_conv/Sqrt:y:0-tfbanks_complex_conv/strided_slice_3:output:0*
T0*
_output_shapes
:(c
tfbanks_complex_conv/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
tfbanks_complex_conv/truedivRealDiv'tfbanks_complex_conv/truediv/x:output:0tfbanks_complex_conv/mul:z:0*
T0*
_output_shapes
:(_
tfbanks_complex_conv/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
tfbanks_complex_conv/powPow-tfbanks_complex_conv/strided_slice_3:output:0#tfbanks_complex_conv/pow/y:output:0*
T0*
_output_shapes
:(a
tfbanks_complex_conv/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
tfbanks_complex_conv/mul_1Mul%tfbanks_complex_conv/mul_1/x:output:0tfbanks_complex_conv/pow:z:0*
T0*
_output_shapes
:(e
 tfbanks_complex_conv/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
tfbanks_complex_conv/truediv_1RealDiv)tfbanks_complex_conv/truediv_1/x:output:0tfbanks_complex_conv/mul_1:z:0*
T0*
_output_shapes
:(a
tfbanks_complex_conv/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
tfbanks_complex_conv/pow_1Pow#tfbanks_complex_conv/range:output:0%tfbanks_complex_conv/pow_1/y:output:0*
T0*
_output_shapes	
:e
tfbanks_complex_conv/NegNegtfbanks_complex_conv/pow_1:z:0*
T0*
_output_shapes	
:}
,tfbanks_complex_conv/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"(      µ
&tfbanks_complex_conv/Tensordot/ReshapeReshape"tfbanks_complex_conv/truediv_1:z:05tfbanks_complex_conv/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:(
.tfbanks_complex_conv/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"     ´
(tfbanks_complex_conv/Tensordot/Reshape_1Reshapetfbanks_complex_conv/Neg:y:07tfbanks_complex_conv/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes
:	½
%tfbanks_complex_conv/Tensordot/MatMulMatMul/tfbanks_complex_conv/Tensordot/Reshape:output:01tfbanks_complex_conv/Tensordot/Reshape_1:output:0*
T0*
_output_shapes
:	(z
tfbanks_complex_conv/ExpExp/tfbanks_complex_conv/Tensordot/MatMul:product:0*
T0*
_output_shapes
:	(
tfbanks_complex_conv/CastCast-tfbanks_complex_conv/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
:(}
tfbanks_complex_conv/Cast_1Cast#tfbanks_complex_conv/range:output:0*

DstT0*

SrcT0*
_output_shapes	
:
.tfbanks_complex_conv/Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"(      ´
(tfbanks_complex_conv/Tensordot_1/ReshapeReshapetfbanks_complex_conv/Cast:y:07tfbanks_complex_conv/Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes

:(
0tfbanks_complex_conv/Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"     »
*tfbanks_complex_conv/Tensordot_1/Reshape_1Reshapetfbanks_complex_conv/Cast_1:y:09tfbanks_complex_conv/Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes
:	Ã
'tfbanks_complex_conv/Tensordot_1/MatMulMatMul1tfbanks_complex_conv/Tensordot_1/Reshape:output:03tfbanks_complex_conv/Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes
:	(e
tfbanks_complex_conv/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB J      ?¥
tfbanks_complex_conv/mul_2Mul%tfbanks_complex_conv/mul_2/x:output:01tfbanks_complex_conv/Tensordot_1/MatMul:product:0*
T0*
_output_shapes
:	(k
tfbanks_complex_conv/Exp_1Exptfbanks_complex_conv/mul_2:z:0*
T0*
_output_shapes
:	(y
tfbanks_complex_conv/Cast_2Cast tfbanks_complex_conv/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
:({
*tfbanks_complex_conv/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        }
,tfbanks_complex_conv/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        }
,tfbanks_complex_conv/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      å
$tfbanks_complex_conv/strided_slice_4StridedSlicetfbanks_complex_conv/Cast_2:y:03tfbanks_complex_conv/strided_slice_4/stack:output:05tfbanks_complex_conv/strided_slice_4/stack_1:output:05tfbanks_complex_conv/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:(*

begin_mask*
end_mask*
new_axis_maskz
tfbanks_complex_conv/Cast_3Casttfbanks_complex_conv/Exp:y:0*

DstT0*

SrcT0*
_output_shapes
:	(
tfbanks_complex_conv/mul_3Mul-tfbanks_complex_conv/strided_slice_4:output:0tfbanks_complex_conv/Exp_1:y:0*
T0*
_output_shapes
:	(
tfbanks_complex_conv/mul_4Multfbanks_complex_conv/mul_3:z:0tfbanks_complex_conv/Cast_3:y:0*
T0*
_output_shapes
:	(b
tfbanks_complex_conv/RealRealtfbanks_complex_conv/mul_4:z:0*
_output_shapes
:	(b
tfbanks_complex_conv/ImagImagtfbanks_complex_conv/mul_4:z:0*
_output_shapes
:	(¯
tfbanks_complex_conv/stack_1Pack"tfbanks_complex_conv/Real:output:0"tfbanks_complex_conv/Imag:output:0*
N*
T0*#
_output_shapes
:(*

axiss
"tfbanks_complex_conv/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"P     ¥
tfbanks_complex_conv/ReshapeReshape%tfbanks_complex_conv/stack_1:output:0+tfbanks_complex_conv/Reshape/shape:output:0*
T0*
_output_shapes
:	Pt
#tfbanks_complex_conv/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       ª
tfbanks_complex_conv/transpose	Transpose%tfbanks_complex_conv/Reshape:output:0,tfbanks_complex_conv/transpose/perm:output:0*
T0*
_output_shapes
:	Pe
#tfbanks_complex_conv/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :­
tfbanks_complex_conv/ExpandDims
ExpandDims"tfbanks_complex_conv/transpose:y:0,tfbanks_complex_conv/ExpandDims/dim:output:0*
T0*#
_output_shapes
:Pu
*tfbanks_complex_conv/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ¼
&tfbanks_complex_conv/conv1d/ExpandDims
ExpandDimsstrided_slice:output:03tfbanks_complex_conv/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}n
,tfbanks_complex_conv/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : É
(tfbanks_complex_conv/conv1d/ExpandDims_1
ExpandDims(tfbanks_complex_conv/ExpandDims:output:05tfbanks_complex_conv/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:Pì
tfbanks_complex_conv/conv1dConv2D/tfbanks_complex_conv/conv1d/ExpandDims:output:01tfbanks_complex_conv/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}P*
paddingSAME*
strides
«
#tfbanks_complex_conv/conv1d/SqueezeSqueeze$tfbanks_complex_conv/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}P*
squeeze_dims

ýÿÿÿÿÿÿÿÿs
squared_modulus/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ´
squared_modulus/transpose	Transpose,tfbanks_complex_conv/conv1d/Squeeze:output:0'squared_modulus/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}Z
squared_modulus/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
squared_modulus/powPowsquared_modulus/transpose:y:0squared_modulus/pow/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}r
0squared_modulus/average_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :É
,squared_modulus/average_pooling1d/ExpandDims
ExpandDimssquared_modulus/pow:z:09squared_modulus/average_pooling1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}â
)squared_modulus/average_pooling1d/AvgPoolAvgPool5squared_modulus/average_pooling1d/ExpandDims:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}*
ksize
*
paddingVALID*
strides
¶
)squared_modulus/average_pooling1d/SqueezeSqueeze2squared_modulus/average_pooling1d/AvgPool:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}*
squeeze_dims
Z
squared_modulus/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @¥
squared_modulus/mulMulsquared_modulus/mul/x:output:02squared_modulus/average_pooling1d/Squeeze:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}u
 squared_modulus/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          £
squared_modulus/transpose_1	Transposesquared_modulus/mul:z:0)squared_modulus/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}(®
.learnable_pooling/clip_by_value/ReadVariableOpReadVariableOp7learnable_pooling_clip_by_value_readvariableop_resource*&
_output_shapes
:(*
dtype0n
)learnable_pooling/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ï
'learnable_pooling/clip_by_value/MinimumMinimum6learnable_pooling/clip_by_value/ReadVariableOp:value:02learnable_pooling/clip_by_value/Minimum/y:output:0*
T0*&
_output_shapes
:(f
!learnable_pooling/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *rn£;´
learnable_pooling/clip_by_valueMaximum+learnable_pooling/clip_by_value/Minimum:z:0*learnable_pooling/clip_by_value/y:output:0*
T0*&
_output_shapes
:(b
learnable_pooling/range/startConst*
_output_shapes
: *
dtype0*
valueB
 *    b
learnable_pooling/range/limitConst*
_output_shapes
: *
dtype0*
valueB
 * ÈCb
learnable_pooling/range/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Á
learnable_pooling/rangeRange&learnable_pooling/range/start:output:0&learnable_pooling/range/limit:output:0&learnable_pooling/range/delta:output:0*

Tidx0*
_output_shapes	
:x
learnable_pooling/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"           ¢
learnable_pooling/ReshapeReshape learnable_pooling/range:output:0(learnable_pooling/Reshape/shape:output:0*
T0*'
_output_shapes
:\
learnable_pooling/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HC
learnable_pooling/subSub"learnable_pooling/Reshape:output:0 learnable_pooling/sub/y:output:0*
T0*'
_output_shapes
:\
learnable_pooling/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
learnable_pooling/mulMul#learnable_pooling/clip_by_value:z:0 learnable_pooling/mul/y:output:0*
T0*&
_output_shapes
:(^
learnable_pooling/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ÈC
learnable_pooling/mul_1Mullearnable_pooling/mul:z:0"learnable_pooling/mul_1/y:output:0*
T0*&
_output_shapes
:(
learnable_pooling/truedivRealDivlearnable_pooling/sub:z:0learnable_pooling/mul_1:z:0*
T0*'
_output_shapes
:(\
learnable_pooling/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
learnable_pooling/powPowlearnable_pooling/truediv:z:0 learnable_pooling/pow/y:output:0*
T0*'
_output_shapes
:(^
learnable_pooling/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ¿
learnable_pooling/mul_2Mul"learnable_pooling/mul_2/x:output:0learnable_pooling/pow:z:0*
T0*'
_output_shapes
:(k
learnable_pooling/ExpExplearnable_pooling/mul_2:z:0*
T0*'
_output_shapes
:(b
 learnable_pooling/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :±
learnable_pooling/ExpandDims
ExpandDimssquared_modulus/transpose_1:y:0)learnable_pooling/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}(z
!learnable_pooling/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"     (      z
)learnable_pooling/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      Ú
learnable_pooling/depthwiseDepthwiseConv2dNative%learnable_pooling/ExpandDims:output:0learnable_pooling/Exp:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(*
paddingSAME*
strides

  
learnable_pooling/SqueezeSqueeze$learnable_pooling/depthwise:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(*
squeeze_dims
N
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7
MaximumMaximum"learnable_pooling/Squeeze:output:0Maximum/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(|
PCEN/Minimum/ReadVariableOpReadVariableOp$pcen_minimum_readvariableop_resource*
_output_shapes
:(*
dtype0S
PCEN/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
PCEN/MinimumMinimum#PCEN/Minimum/ReadVariableOp:value:0PCEN/Minimum/y:output:0*
T0*
_output_shapes
:(|
PCEN/Maximum/ReadVariableOpReadVariableOp$pcen_maximum_readvariableop_resource*
_output_shapes
:(*
dtype0S
PCEN/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
PCEN/MaximumMaximum#PCEN/Maximum/ReadVariableOp:value:0PCEN/Maximum/y:output:0*
T0*
_output_shapes
:(W
PCEN/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B : T
PCEN/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :¸
PCEN/GatherV2GatherV2Maximum:z:0PCEN/GatherV2/indices:output:0PCEN/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
%PCEN/EMA/clip_by_value/ReadVariableOpReadVariableOp.pcen_ema_clip_by_value_readvariableop_resource*
_output_shapes
:(*
dtype0e
 PCEN/EMA/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¨
PCEN/EMA/clip_by_value/MinimumMinimum-PCEN/EMA/clip_by_value/ReadVariableOp:value:0)PCEN/EMA/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:(]
PCEN/EMA/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
PCEN/EMA/clip_by_valueMaximum"PCEN/EMA/clip_by_value/Minimum:z:0!PCEN/EMA/clip_by_value/y:output:0*
T0*
_output_shapes
:(l
PCEN/EMA/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
PCEN/EMA/transpose	TransposeMaximum:z:0 PCEN/EMA/transpose/perm:output:0*
T0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿ(z
)PCEN/EMA/scan/TensorArrayV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   j
(PCEN/EMA/scan/TensorArrayV2/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :dé
PCEN/EMA/scan/TensorArrayV2TensorListReserve2PCEN/EMA/scan/TensorArrayV2/element_shape:output:01PCEN/EMA/scan/TensorArrayV2/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
CPCEN/EMA/scan/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   
5PCEN/EMA/scan/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorPCEN/EMA/transpose:y:0LPCEN/EMA/scan/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ|
+PCEN/EMA/scan/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   l
*PCEN/EMA/scan/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :dï
PCEN/EMA/scan/TensorArrayV2_1TensorListReserve4PCEN/EMA/scan/TensorArrayV2_1/element_shape:output:03PCEN/EMA/scan/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒU
PCEN/EMA/scan/ConstConst*
_output_shapes
: *
dtype0*
value	B : h
&PCEN/EMA/scan/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
value	B :db
 PCEN/EMA/scan/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
PCEN/EMA/scan/whileStatelessWhile)PCEN/EMA/scan/while/loop_counter:output:0/PCEN/EMA/scan/while/maximum_iterations:output:0PCEN/EMA/scan/Const:output:0PCEN/GatherV2:output:0&PCEN/EMA/scan/TensorArrayV2_1:handle:0EPCEN/EMA/scan/TensorArrayUnstack/TensorListFromTensor:output_handle:0PCEN/EMA/clip_by_value:z:0*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*7
_output_shapes%
#: : : :ÿÿÿÿÿÿÿÿÿ(: : :(* 
_read_only_resource_inputs
 *
_stateful_parallelism( **
body"R 
PCEN_EMA_scan_while_body_15208**
cond"R 
PCEN_EMA_scan_while_cond_15207*6
output_shapes%
#: : : :ÿÿÿÿÿÿÿÿÿ(: : :(
>PCEN/EMA/scan/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   
0PCEN/EMA/scan/TensorArrayV2Stack/TensorListStackTensorListStackPCEN/EMA/scan/while:output:4GPCEN/EMA/scan/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿ(*
element_dtype0*
num_elementsdn
PCEN/EMA/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¶
PCEN/EMA/transpose_1	Transpose9PCEN/EMA/scan/TensorArrayV2Stack/TensorListStack:tensor:0"PCEN/EMA/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(S
PCEN/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?g
PCEN/truedivRealDivPCEN/truediv/x:output:0PCEN/Maximum:z:0*
T0*
_output_shapes
:(O

PCEN/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ì¼+v
PCEN/addAddV2PCEN/add/x:output:0PCEN/EMA/transpose_1:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(e
PCEN/powPowPCEN/add:z:0PCEN/Minimum:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(j
PCEN/truediv_1RealDivMaximum:z:0PCEN/pow:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(x
PCEN/add_1/ReadVariableOpReadVariableOp"pcen_add_1_readvariableop_resource*
_output_shapes
:(*
dtype0

PCEN/add_1AddV2PCEN/truediv_1:z:0!PCEN/add_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(i

PCEN/pow_1PowPCEN/add_1:z:0PCEN/truediv:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(r
PCEN/ReadVariableOpReadVariableOp"pcen_add_1_readvariableop_resource*
_output_shapes
:(*
dtype0e

PCEN/pow_2PowPCEN/ReadVariableOp:value:0PCEN/truediv:z:0*
T0*
_output_shapes
:(e
PCEN/subSubPCEN/pow_1:z:0PCEN/pow_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(X
IdentityIdentityPCEN/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(f

Identity_1IdentityIdentity:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(Û
NoOpNoOp&^PCEN/EMA/clip_by_value/ReadVariableOp^PCEN/Maximum/ReadVariableOp^PCEN/Minimum/ReadVariableOp^PCEN/ReadVariableOp^PCEN/add_1/ReadVariableOp/^learnable_pooling/clip_by_value/ReadVariableOp$^tfbanks_complex_conv/ReadVariableOp&^tfbanks_complex_conv/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ}: : : : : : 2N
%PCEN/EMA/clip_by_value/ReadVariableOp%PCEN/EMA/clip_by_value/ReadVariableOp2:
PCEN/Maximum/ReadVariableOpPCEN/Maximum/ReadVariableOp2:
PCEN/Minimum/ReadVariableOpPCEN/Minimum/ReadVariableOp2*
PCEN/ReadVariableOpPCEN/ReadVariableOp26
PCEN/add_1/ReadVariableOpPCEN/add_1/ReadVariableOp2`
.learnable_pooling/clip_by_value/ReadVariableOp.learnable_pooling/clip_by_value/ReadVariableOp2J
#tfbanks_complex_conv/ReadVariableOp#tfbanks_complex_conv/ReadVariableOp2N
%tfbanks_complex_conv/ReadVariableOp_1%tfbanks_complex_conv/ReadVariableOp_1:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
 
_user_specified_nameinputs
å


EMA_scan_while_cond_13670.
*ema_scan_while_ema_scan_while_loop_counter4
0ema_scan_while_ema_scan_while_maximum_iterations
ema_scan_while_placeholder 
ema_scan_while_placeholder_1 
ema_scan_while_placeholder_2E
Aema_scan_while_ema_scan_while_cond_13670___redundant_placeholder0E
Aema_scan_while_ema_scan_while_cond_13670___redundant_placeholder1
ema_scan_while_identity
W
EMA/scan/while/Less/yConst*
_output_shapes
: *
dtype0*
value	B :dx
EMA/scan/while/LessLessema_scan_while_placeholderEMA/scan/while/Less/y:output:0*
T0*
_output_shapes
: 
EMA/scan/while/Less_1Less*ema_scan_while_ema_scan_while_loop_counter0ema_scan_while_ema_scan_while_maximum_iterations*
T0*
_output_shapes
: s
EMA/scan/while/LogicalAnd
LogicalAndEMA/scan/while/Less_1:z:0EMA/scan/while/Less:z:0*
_output_shapes
: c
EMA/scan/while/IdentityIdentityEMA/scan/while/LogicalAnd:z:0*
T0
*
_output_shapes
: ";
ema_scan_while_identity EMA/scan/while/Identity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : :ÿÿÿÿÿÿÿÿÿ(: ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:

_output_shapes
: :

_output_shapes
::

_output_shapes
:
1
ã	
4audio_classifier_leaf_PCEN_EMA_scan_while_body_13384d
`audio_classifier_leaf_pcen_ema_scan_while_audio_classifier_leaf_pcen_ema_scan_while_loop_counterj
faudio_classifier_leaf_pcen_ema_scan_while_audio_classifier_leaf_pcen_ema_scan_while_maximum_iterations9
5audio_classifier_leaf_pcen_ema_scan_while_placeholder;
7audio_classifier_leaf_pcen_ema_scan_while_placeholder_1;
7audio_classifier_leaf_pcen_ema_scan_while_placeholder_2 
audio_classifier_leaf_pcen_ema_scan_while_tensorarrayv2read_tensorlistgetitem_audio_classifier_leaf_pcen_ema_scan_tensorarrayunstack_tensorlistfromtensor_0`
\audio_classifier_leaf_pcen_ema_scan_while_mul_audio_classifier_leaf_pcen_ema_clip_by_value_06
2audio_classifier_leaf_pcen_ema_scan_while_identity8
4audio_classifier_leaf_pcen_ema_scan_while_identity_18
4audio_classifier_leaf_pcen_ema_scan_while_identity_28
4audio_classifier_leaf_pcen_ema_scan_while_identity_38
4audio_classifier_leaf_pcen_ema_scan_while_identity_4
audio_classifier_leaf_pcen_ema_scan_while_tensorarrayv2read_tensorlistgetitem_audio_classifier_leaf_pcen_ema_scan_tensorarrayunstack_tensorlistfromtensor^
Zaudio_classifier_leaf_pcen_ema_scan_while_mul_audio_classifier_leaf_pcen_ema_clip_by_value¬
[audio_classifier/leaf/PCEN/EMA/scan/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   Û
Maudio_classifier/leaf/PCEN/EMA/scan/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemaudio_classifier_leaf_pcen_ema_scan_while_tensorarrayv2read_tensorlistgetitem_audio_classifier_leaf_pcen_ema_scan_tensorarrayunstack_tensorlistfromtensor_05audio_classifier_leaf_pcen_ema_scan_while_placeholderdaudio_classifier/leaf/PCEN/EMA/scan/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
element_dtype0
-audio_classifier/leaf/PCEN/EMA/scan/while/mulMul\audio_classifier_leaf_pcen_ema_scan_while_mul_audio_classifier_leaf_pcen_ema_clip_by_value_0Taudio_classifier/leaf/PCEN/EMA/scan/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(t
/audio_classifier/leaf/PCEN/EMA/scan/while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ñ
-audio_classifier/leaf/PCEN/EMA/scan/while/subSub8audio_classifier/leaf/PCEN/EMA/scan/while/sub/x:output:0\audio_classifier_leaf_pcen_ema_scan_while_mul_audio_classifier_leaf_pcen_ema_clip_by_value_0*
T0*
_output_shapes
:(Ô
/audio_classifier/leaf/PCEN/EMA/scan/while/mul_1Mul1audio_classifier/leaf/PCEN/EMA/scan/while/sub:z:07audio_classifier_leaf_pcen_ema_scan_while_placeholder_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Ð
-audio_classifier/leaf/PCEN/EMA/scan/while/addAddV21audio_classifier/leaf/PCEN/EMA/scan/while/mul:z:03audio_classifier/leaf/PCEN/EMA/scan/while/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Æ
Naudio_classifier/leaf/PCEN/EMA/scan/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem7audio_classifier_leaf_pcen_ema_scan_while_placeholder_25audio_classifier_leaf_pcen_ema_scan_while_placeholder1audio_classifier/leaf/PCEN/EMA/scan/while/add:z:0*
_output_shapes
: *
element_dtype0:éèÒs
1audio_classifier/leaf/PCEN/EMA/scan/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ì
/audio_classifier/leaf/PCEN/EMA/scan/while/add_1AddV25audio_classifier_leaf_pcen_ema_scan_while_placeholder:audio_classifier/leaf/PCEN/EMA/scan/while/add_1/y:output:0*
T0*
_output_shapes
: s
1audio_classifier/leaf/PCEN/EMA/scan/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :÷
/audio_classifier/leaf/PCEN/EMA/scan/while/add_2AddV2`audio_classifier_leaf_pcen_ema_scan_while_audio_classifier_leaf_pcen_ema_scan_while_loop_counter:audio_classifier/leaf/PCEN/EMA/scan/while/add_2/y:output:0*
T0*
_output_shapes
: 
2audio_classifier/leaf/PCEN/EMA/scan/while/IdentityIdentity3audio_classifier/leaf/PCEN/EMA/scan/while/add_2:z:0*
T0*
_output_shapes
: É
4audio_classifier/leaf/PCEN/EMA/scan/while/Identity_1Identityfaudio_classifier_leaf_pcen_ema_scan_while_audio_classifier_leaf_pcen_ema_scan_while_maximum_iterations*
T0*
_output_shapes
: 
4audio_classifier/leaf/PCEN/EMA/scan/while/Identity_2Identity3audio_classifier/leaf/PCEN/EMA/scan/while/add_1:z:0*
T0*
_output_shapes
: ¥
4audio_classifier/leaf/PCEN/EMA/scan/while/Identity_3Identity1audio_classifier/leaf/PCEN/EMA/scan/while/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Á
4audio_classifier/leaf/PCEN/EMA/scan/while/Identity_4Identity^audio_classifier/leaf/PCEN/EMA/scan/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "q
2audio_classifier_leaf_pcen_ema_scan_while_identity;audio_classifier/leaf/PCEN/EMA/scan/while/Identity:output:0"u
4audio_classifier_leaf_pcen_ema_scan_while_identity_1=audio_classifier/leaf/PCEN/EMA/scan/while/Identity_1:output:0"u
4audio_classifier_leaf_pcen_ema_scan_while_identity_2=audio_classifier/leaf/PCEN/EMA/scan/while/Identity_2:output:0"u
4audio_classifier_leaf_pcen_ema_scan_while_identity_3=audio_classifier/leaf/PCEN/EMA/scan/while/Identity_3:output:0"u
4audio_classifier_leaf_pcen_ema_scan_while_identity_4=audio_classifier/leaf/PCEN/EMA/scan/while/Identity_4:output:0"º
Zaudio_classifier_leaf_pcen_ema_scan_while_mul_audio_classifier_leaf_pcen_ema_clip_by_value\audio_classifier_leaf_pcen_ema_scan_while_mul_audio_classifier_leaf_pcen_ema_clip_by_value_0"º
audio_classifier_leaf_pcen_ema_scan_while_tensorarrayv2read_tensorlistgetitem_audio_classifier_leaf_pcen_ema_scan_tensorarrayunstack_tensorlistfromtensoraudio_classifier_leaf_pcen_ema_scan_while_tensorarrayv2read_tensorlistgetitem_audio_classifier_leaf_pcen_ema_scan_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : :ÿÿÿÿÿÿÿÿÿ(: : :(: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:(

¡
?__inference_leaf_layer_call_and_return_conditional_losses_13748

inputs,
tfbanks_complex_conv_13578:(1
learnable_pooling_13634:(

pcen_13738:(

pcen_13740:(

pcen_13742:(

pcen_13744:(
identity¢PCEN/StatefulPartitionedCall¢)learnable_pooling/StatefulPartitionedCall¢,tfbanks_complex_conv/StatefulPartitionedCallh
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         þ
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*

begin_mask*
end_mask*
new_axis_mask
,tfbanks_complex_conv/StatefulPartitionedCallStatefulPartitionedCallstrided_slice:output:0tfbanks_complex_conv_13578*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}P*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_tfbanks_complex_conv_layer_call_and_return_conditional_losses_13577ü
squared_modulus/PartitionedCallPartitionedCall5tfbanks_complex_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_squared_modulus_layer_call_and_return_conditional_losses_13597
)learnable_pooling/StatefulPartitionedCallStatefulPartitionedCall(squared_modulus/PartitionedCall:output:0learnable_pooling_13634*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_learnable_pooling_layer_call_and_return_conditional_losses_13633N
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7
MaximumMaximum2learnable_pooling/StatefulPartitionedCall:output:0Maximum/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(
PCEN/StatefulPartitionedCallStatefulPartitionedCallMaximum:z:0
pcen_13738
pcen_13740
pcen_13742
pcen_13744*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_PCEN_layer_call_and_return_conditional_losses_13737x
IdentityIdentity%PCEN/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(À
NoOpNoOp^PCEN/StatefulPartitionedCall*^learnable_pooling/StatefulPartitionedCall-^tfbanks_complex_conv/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ}: : : : : : 2<
PCEN/StatefulPartitionedCallPCEN/StatefulPartitionedCall2V
)learnable_pooling/StatefulPartitionedCall)learnable_pooling/StatefulPartitionedCall2\
,tfbanks_complex_conv/StatefulPartitionedCall,tfbanks_complex_conv/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
 
_user_specified_nameinputs
å


EMA_scan_while_cond_15522.
*ema_scan_while_ema_scan_while_loop_counter4
0ema_scan_while_ema_scan_while_maximum_iterations
ema_scan_while_placeholder 
ema_scan_while_placeholder_1 
ema_scan_while_placeholder_2E
Aema_scan_while_ema_scan_while_cond_15522___redundant_placeholder0E
Aema_scan_while_ema_scan_while_cond_15522___redundant_placeholder1
ema_scan_while_identity
W
EMA/scan/while/Less/yConst*
_output_shapes
: *
dtype0*
value	B :dx
EMA/scan/while/LessLessema_scan_while_placeholderEMA/scan/while/Less/y:output:0*
T0*
_output_shapes
: 
EMA/scan/while/Less_1Less*ema_scan_while_ema_scan_while_loop_counter0ema_scan_while_ema_scan_while_maximum_iterations*
T0*
_output_shapes
: s
EMA/scan/while/LogicalAnd
LogicalAndEMA/scan/while/Less_1:z:0EMA/scan/while/Less:z:0*
_output_shapes
: c
EMA/scan/while/IdentityIdentityEMA/scan/while/LogicalAnd:z:0*
T0
*
_output_shapes
: ";
ema_scan_while_identity EMA/scan/while/Identity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : :ÿÿÿÿÿÿÿÿÿ(: ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:

_output_shapes
: :

_output_shapes
::

_output_shapes
:
Ç

4__inference_tfbanks_complex_conv_layer_call_fn_15327

inputs
unknown:(
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}P*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_tfbanks_complex_conv_layer_call_and_return_conditional_losses_13577t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:ÿÿÿÿÿÿÿÿÿ}: 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
 
_user_specified_nameinputs
Ã	
ñ
@__inference_dense_layer_call_and_return_conditional_losses_15320

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²
^
B__inference_flatten_layer_call_and_return_conditional_losses_13963

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
¨
EMA_scan_while_body_13671.
*ema_scan_while_ema_scan_while_loop_counter4
0ema_scan_while_ema_scan_while_maximum_iterations
ema_scan_while_placeholder 
ema_scan_while_placeholder_1 
ema_scan_while_placeholder_2i
eema_scan_while_tensorarrayv2read_tensorlistgetitem_ema_scan_tensorarrayunstack_tensorlistfromtensor_0*
&ema_scan_while_mul_ema_clip_by_value_0
ema_scan_while_identity
ema_scan_while_identity_1
ema_scan_while_identity_2
ema_scan_while_identity_3
ema_scan_while_identity_4g
cema_scan_while_tensorarrayv2read_tensorlistgetitem_ema_scan_tensorarrayunstack_tensorlistfromtensor(
$ema_scan_while_mul_ema_clip_by_value
@EMA/scan/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   Ó
2EMA/scan/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemeema_scan_while_tensorarrayv2read_tensorlistgetitem_ema_scan_tensorarrayunstack_tensorlistfromtensor_0ema_scan_while_placeholderIEMA/scan/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
element_dtype0®
EMA/scan/while/mulMul&ema_scan_while_mul_ema_clip_by_value_09EMA/scan/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Y
EMA/scan/while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
EMA/scan/while/subSubEMA/scan/while/sub/x:output:0&ema_scan_while_mul_ema_clip_by_value_0*
T0*
_output_shapes
:(
EMA/scan/while/mul_1MulEMA/scan/while/sub:z:0ema_scan_while_placeholder_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
EMA/scan/while/addAddV2EMA/scan/while/mul:z:0EMA/scan/while/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(Ú
3EMA/scan/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemema_scan_while_placeholder_2ema_scan_while_placeholderEMA/scan/while/add:z:0*
_output_shapes
: *
element_dtype0:éèÒX
EMA/scan/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :{
EMA/scan/while/add_1AddV2ema_scan_while_placeholderEMA/scan/while/add_1/y:output:0*
T0*
_output_shapes
: X
EMA/scan/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :
EMA/scan/while/add_2AddV2*ema_scan_while_ema_scan_while_loop_counterEMA/scan/while/add_2/y:output:0*
T0*
_output_shapes
: ^
EMA/scan/while/IdentityIdentityEMA/scan/while/add_2:z:0*
T0*
_output_shapes
: x
EMA/scan/while/Identity_1Identity0ema_scan_while_ema_scan_while_maximum_iterations*
T0*
_output_shapes
: `
EMA/scan/while/Identity_2IdentityEMA/scan/while/add_1:z:0*
T0*
_output_shapes
: o
EMA/scan/while/Identity_3IdentityEMA/scan/while/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
EMA/scan/while/Identity_4IdentityCEMA/scan/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: ";
ema_scan_while_identity EMA/scan/while/Identity:output:0"?
ema_scan_while_identity_1"EMA/scan/while/Identity_1:output:0"?
ema_scan_while_identity_2"EMA/scan/while/Identity_2:output:0"?
ema_scan_while_identity_3"EMA/scan/while/Identity_3:output:0"?
ema_scan_while_identity_4"EMA/scan/while/Identity_4:output:0"N
$ema_scan_while_mul_ema_clip_by_value&ema_scan_while_mul_ema_clip_by_value_0"Ì
cema_scan_while_tensorarrayv2read_tensorlistgetitem_ema_scan_tensorarrayunstack_tensorlistfromtensoreema_scan_while_tensorarrayv2read_tensorlistgetitem_ema_scan_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : :ÿÿÿÿÿÿÿÿÿ(: : :(: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:(

a
J__inference_squared_modulus_layer_call_and_return_conditional_losses_13597
x
identityc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          i
	transpose	Transposextranspose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @`
powPowtranspose:y:0pow/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}b
 average_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :
average_pooling1d/ExpandDims
ExpandDimspow:z:0)average_pooling1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}Â
average_pooling1d/AvgPoolAvgPool%average_pooling1d/ExpandDims:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}*
ksize
*
paddingVALID*
strides

average_pooling1d/SqueezeSqueeze"average_pooling1d/AvgPool:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}*
squeeze_dims
J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @u
mulMulmul/x:output:0"average_pooling1d/Squeeze:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}e
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          s
transpose_1	Transposemul:z:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}(\
IdentityIdentitytranspose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ}P:O K
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}P

_user_specified_namex
Æ×
Þ
?__inference_leaf_layer_call_and_return_conditional_losses_15043

inputs>
,tfbanks_complex_conv_readvariableop_resource:(Q
7learnable_pooling_clip_by_value_readvariableop_resource:(2
$pcen_minimum_readvariableop_resource:(2
$pcen_maximum_readvariableop_resource:(<
.pcen_ema_clip_by_value_readvariableop_resource:(0
"pcen_add_1_readvariableop_resource:(
identity¢%PCEN/EMA/clip_by_value/ReadVariableOp¢PCEN/Maximum/ReadVariableOp¢PCEN/Minimum/ReadVariableOp¢PCEN/ReadVariableOp¢PCEN/add_1/ReadVariableOp¢.learnable_pooling/clip_by_value/ReadVariableOp¢#tfbanks_complex_conv/ReadVariableOp¢%tfbanks_complex_conv/ReadVariableOp_1h
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         þ
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*

begin_mask*
end_mask*
new_axis_mask
#tfbanks_complex_conv/ReadVariableOpReadVariableOp,tfbanks_complex_conv_readvariableop_resource*
_output_shapes

:(*
dtype0y
(tfbanks_complex_conv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        {
*tfbanks_complex_conv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*tfbanks_complex_conv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      è
"tfbanks_complex_conv/strided_sliceStridedSlice+tfbanks_complex_conv/ReadVariableOp:value:01tfbanks_complex_conv/strided_slice/stack:output:03tfbanks_complex_conv/strided_slice/stack_1:output:03tfbanks_complex_conv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:(*

begin_mask*
end_mask*
shrink_axis_maskq
,tfbanks_complex_conv/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÛI@¾
*tfbanks_complex_conv/clip_by_value/MinimumMinimum+tfbanks_complex_conv/strided_slice:output:05tfbanks_complex_conv/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:(i
$tfbanks_complex_conv/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    ±
"tfbanks_complex_conv/clip_by_valueMaximum.tfbanks_complex_conv/clip_by_value/Minimum:z:0-tfbanks_complex_conv/clip_by_value/y:output:0*
T0*
_output_shapes
:(
%tfbanks_complex_conv/ReadVariableOp_1ReadVariableOp,tfbanks_complex_conv_readvariableop_resource*
_output_shapes

:(*
dtype0{
*tfbanks_complex_conv/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,tfbanks_complex_conv/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,tfbanks_complex_conv/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ò
$tfbanks_complex_conv/strided_slice_1StridedSlice-tfbanks_complex_conv/ReadVariableOp_1:value:03tfbanks_complex_conv/strided_slice_1/stack:output:05tfbanks_complex_conv/strided_slice_1/stack_1:output:05tfbanks_complex_conv/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:(*

begin_mask*
end_mask*
shrink_axis_masks
.tfbanks_complex_conv/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ICÄ
,tfbanks_complex_conv/clip_by_value_1/MinimumMinimum-tfbanks_complex_conv/strided_slice_1:output:07tfbanks_complex_conv/clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:(k
&tfbanks_complex_conv/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *Tã¿?·
$tfbanks_complex_conv/clip_by_value_1Maximum0tfbanks_complex_conv/clip_by_value_1/Minimum:z:0/tfbanks_complex_conv/clip_by_value_1/y:output:0*
T0*
_output_shapes
:(²
tfbanks_complex_conv/stackPack&tfbanks_complex_conv/clip_by_value:z:0(tfbanks_complex_conv/clip_by_value_1:z:0*
N*
T0*
_output_shapes

:(*

axise
 tfbanks_complex_conv/range/startConst*
_output_shapes
: *
dtype0*
valueB
 *  HÃe
 tfbanks_complex_conv/range/limitConst*
_output_shapes
: *
dtype0*
valueB
 *  ICe
 tfbanks_complex_conv/range/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Í
tfbanks_complex_conv/rangeRange)tfbanks_complex_conv/range/start:output:0)tfbanks_complex_conv/range/limit:output:0)tfbanks_complex_conv/range/delta:output:0*

Tidx0*
_output_shapes	
:{
*tfbanks_complex_conv/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        }
,tfbanks_complex_conv/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,tfbanks_complex_conv/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      è
$tfbanks_complex_conv/strided_slice_2StridedSlice#tfbanks_complex_conv/stack:output:03tfbanks_complex_conv/strided_slice_2/stack:output:05tfbanks_complex_conv/strided_slice_2/stack_1:output:05tfbanks_complex_conv/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:(*

begin_mask*
end_mask*
shrink_axis_mask{
*tfbanks_complex_conv/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,tfbanks_complex_conv/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,tfbanks_complex_conv/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      è
$tfbanks_complex_conv/strided_slice_3StridedSlice#tfbanks_complex_conv/stack:output:03tfbanks_complex_conv/strided_slice_3/stack:output:05tfbanks_complex_conv/strided_slice_3/stack_1:output:05tfbanks_complex_conv/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:(*

begin_mask*
end_mask*
shrink_axis_mask`
tfbanks_complex_conv/Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÛÉ@h
tfbanks_complex_conv/SqrtSqrt$tfbanks_complex_conv/Sqrt/x:output:0*
T0*
_output_shapes
: 
tfbanks_complex_conv/mulMultfbanks_complex_conv/Sqrt:y:0-tfbanks_complex_conv/strided_slice_3:output:0*
T0*
_output_shapes
:(c
tfbanks_complex_conv/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
tfbanks_complex_conv/truedivRealDiv'tfbanks_complex_conv/truediv/x:output:0tfbanks_complex_conv/mul:z:0*
T0*
_output_shapes
:(_
tfbanks_complex_conv/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
tfbanks_complex_conv/powPow-tfbanks_complex_conv/strided_slice_3:output:0#tfbanks_complex_conv/pow/y:output:0*
T0*
_output_shapes
:(a
tfbanks_complex_conv/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
tfbanks_complex_conv/mul_1Mul%tfbanks_complex_conv/mul_1/x:output:0tfbanks_complex_conv/pow:z:0*
T0*
_output_shapes
:(e
 tfbanks_complex_conv/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
tfbanks_complex_conv/truediv_1RealDiv)tfbanks_complex_conv/truediv_1/x:output:0tfbanks_complex_conv/mul_1:z:0*
T0*
_output_shapes
:(a
tfbanks_complex_conv/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
tfbanks_complex_conv/pow_1Pow#tfbanks_complex_conv/range:output:0%tfbanks_complex_conv/pow_1/y:output:0*
T0*
_output_shapes	
:e
tfbanks_complex_conv/NegNegtfbanks_complex_conv/pow_1:z:0*
T0*
_output_shapes	
:}
,tfbanks_complex_conv/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"(      µ
&tfbanks_complex_conv/Tensordot/ReshapeReshape"tfbanks_complex_conv/truediv_1:z:05tfbanks_complex_conv/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:(
.tfbanks_complex_conv/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"     ´
(tfbanks_complex_conv/Tensordot/Reshape_1Reshapetfbanks_complex_conv/Neg:y:07tfbanks_complex_conv/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes
:	½
%tfbanks_complex_conv/Tensordot/MatMulMatMul/tfbanks_complex_conv/Tensordot/Reshape:output:01tfbanks_complex_conv/Tensordot/Reshape_1:output:0*
T0*
_output_shapes
:	(z
tfbanks_complex_conv/ExpExp/tfbanks_complex_conv/Tensordot/MatMul:product:0*
T0*
_output_shapes
:	(
tfbanks_complex_conv/CastCast-tfbanks_complex_conv/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
:(}
tfbanks_complex_conv/Cast_1Cast#tfbanks_complex_conv/range:output:0*

DstT0*

SrcT0*
_output_shapes	
:
.tfbanks_complex_conv/Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"(      ´
(tfbanks_complex_conv/Tensordot_1/ReshapeReshapetfbanks_complex_conv/Cast:y:07tfbanks_complex_conv/Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes

:(
0tfbanks_complex_conv/Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"     »
*tfbanks_complex_conv/Tensordot_1/Reshape_1Reshapetfbanks_complex_conv/Cast_1:y:09tfbanks_complex_conv/Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes
:	Ã
'tfbanks_complex_conv/Tensordot_1/MatMulMatMul1tfbanks_complex_conv/Tensordot_1/Reshape:output:03tfbanks_complex_conv/Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes
:	(e
tfbanks_complex_conv/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB J      ?¥
tfbanks_complex_conv/mul_2Mul%tfbanks_complex_conv/mul_2/x:output:01tfbanks_complex_conv/Tensordot_1/MatMul:product:0*
T0*
_output_shapes
:	(k
tfbanks_complex_conv/Exp_1Exptfbanks_complex_conv/mul_2:z:0*
T0*
_output_shapes
:	(y
tfbanks_complex_conv/Cast_2Cast tfbanks_complex_conv/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
:({
*tfbanks_complex_conv/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        }
,tfbanks_complex_conv/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        }
,tfbanks_complex_conv/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      å
$tfbanks_complex_conv/strided_slice_4StridedSlicetfbanks_complex_conv/Cast_2:y:03tfbanks_complex_conv/strided_slice_4/stack:output:05tfbanks_complex_conv/strided_slice_4/stack_1:output:05tfbanks_complex_conv/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:(*

begin_mask*
end_mask*
new_axis_maskz
tfbanks_complex_conv/Cast_3Casttfbanks_complex_conv/Exp:y:0*

DstT0*

SrcT0*
_output_shapes
:	(
tfbanks_complex_conv/mul_3Mul-tfbanks_complex_conv/strided_slice_4:output:0tfbanks_complex_conv/Exp_1:y:0*
T0*
_output_shapes
:	(
tfbanks_complex_conv/mul_4Multfbanks_complex_conv/mul_3:z:0tfbanks_complex_conv/Cast_3:y:0*
T0*
_output_shapes
:	(b
tfbanks_complex_conv/RealRealtfbanks_complex_conv/mul_4:z:0*
_output_shapes
:	(b
tfbanks_complex_conv/ImagImagtfbanks_complex_conv/mul_4:z:0*
_output_shapes
:	(¯
tfbanks_complex_conv/stack_1Pack"tfbanks_complex_conv/Real:output:0"tfbanks_complex_conv/Imag:output:0*
N*
T0*#
_output_shapes
:(*

axiss
"tfbanks_complex_conv/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"P     ¥
tfbanks_complex_conv/ReshapeReshape%tfbanks_complex_conv/stack_1:output:0+tfbanks_complex_conv/Reshape/shape:output:0*
T0*
_output_shapes
:	Pt
#tfbanks_complex_conv/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       ª
tfbanks_complex_conv/transpose	Transpose%tfbanks_complex_conv/Reshape:output:0,tfbanks_complex_conv/transpose/perm:output:0*
T0*
_output_shapes
:	Pe
#tfbanks_complex_conv/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :­
tfbanks_complex_conv/ExpandDims
ExpandDims"tfbanks_complex_conv/transpose:y:0,tfbanks_complex_conv/ExpandDims/dim:output:0*
T0*#
_output_shapes
:Pu
*tfbanks_complex_conv/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ¼
&tfbanks_complex_conv/conv1d/ExpandDims
ExpandDimsstrided_slice:output:03tfbanks_complex_conv/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}n
,tfbanks_complex_conv/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : É
(tfbanks_complex_conv/conv1d/ExpandDims_1
ExpandDims(tfbanks_complex_conv/ExpandDims:output:05tfbanks_complex_conv/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:Pì
tfbanks_complex_conv/conv1dConv2D/tfbanks_complex_conv/conv1d/ExpandDims:output:01tfbanks_complex_conv/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}P*
paddingSAME*
strides
«
#tfbanks_complex_conv/conv1d/SqueezeSqueeze$tfbanks_complex_conv/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}P*
squeeze_dims

ýÿÿÿÿÿÿÿÿs
squared_modulus/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ´
squared_modulus/transpose	Transpose,tfbanks_complex_conv/conv1d/Squeeze:output:0'squared_modulus/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}Z
squared_modulus/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
squared_modulus/powPowsquared_modulus/transpose:y:0squared_modulus/pow/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}r
0squared_modulus/average_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :É
,squared_modulus/average_pooling1d/ExpandDims
ExpandDimssquared_modulus/pow:z:09squared_modulus/average_pooling1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}â
)squared_modulus/average_pooling1d/AvgPoolAvgPool5squared_modulus/average_pooling1d/ExpandDims:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}*
ksize
*
paddingVALID*
strides
¶
)squared_modulus/average_pooling1d/SqueezeSqueeze2squared_modulus/average_pooling1d/AvgPool:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}*
squeeze_dims
Z
squared_modulus/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @¥
squared_modulus/mulMulsquared_modulus/mul/x:output:02squared_modulus/average_pooling1d/Squeeze:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}u
 squared_modulus/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          £
squared_modulus/transpose_1	Transposesquared_modulus/mul:z:0)squared_modulus/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}(®
.learnable_pooling/clip_by_value/ReadVariableOpReadVariableOp7learnable_pooling_clip_by_value_readvariableop_resource*&
_output_shapes
:(*
dtype0n
)learnable_pooling/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ï
'learnable_pooling/clip_by_value/MinimumMinimum6learnable_pooling/clip_by_value/ReadVariableOp:value:02learnable_pooling/clip_by_value/Minimum/y:output:0*
T0*&
_output_shapes
:(f
!learnable_pooling/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *rn£;´
learnable_pooling/clip_by_valueMaximum+learnable_pooling/clip_by_value/Minimum:z:0*learnable_pooling/clip_by_value/y:output:0*
T0*&
_output_shapes
:(b
learnable_pooling/range/startConst*
_output_shapes
: *
dtype0*
valueB
 *    b
learnable_pooling/range/limitConst*
_output_shapes
: *
dtype0*
valueB
 * ÈCb
learnable_pooling/range/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Á
learnable_pooling/rangeRange&learnable_pooling/range/start:output:0&learnable_pooling/range/limit:output:0&learnable_pooling/range/delta:output:0*

Tidx0*
_output_shapes	
:x
learnable_pooling/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"           ¢
learnable_pooling/ReshapeReshape learnable_pooling/range:output:0(learnable_pooling/Reshape/shape:output:0*
T0*'
_output_shapes
:\
learnable_pooling/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HC
learnable_pooling/subSub"learnable_pooling/Reshape:output:0 learnable_pooling/sub/y:output:0*
T0*'
_output_shapes
:\
learnable_pooling/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
learnable_pooling/mulMul#learnable_pooling/clip_by_value:z:0 learnable_pooling/mul/y:output:0*
T0*&
_output_shapes
:(^
learnable_pooling/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ÈC
learnable_pooling/mul_1Mullearnable_pooling/mul:z:0"learnable_pooling/mul_1/y:output:0*
T0*&
_output_shapes
:(
learnable_pooling/truedivRealDivlearnable_pooling/sub:z:0learnable_pooling/mul_1:z:0*
T0*'
_output_shapes
:(\
learnable_pooling/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
learnable_pooling/powPowlearnable_pooling/truediv:z:0 learnable_pooling/pow/y:output:0*
T0*'
_output_shapes
:(^
learnable_pooling/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ¿
learnable_pooling/mul_2Mul"learnable_pooling/mul_2/x:output:0learnable_pooling/pow:z:0*
T0*'
_output_shapes
:(k
learnable_pooling/ExpExplearnable_pooling/mul_2:z:0*
T0*'
_output_shapes
:(b
 learnable_pooling/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :±
learnable_pooling/ExpandDims
ExpandDimssquared_modulus/transpose_1:y:0)learnable_pooling/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}(z
!learnable_pooling/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"     (      z
)learnable_pooling/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      Ú
learnable_pooling/depthwiseDepthwiseConv2dNative%learnable_pooling/ExpandDims:output:0learnable_pooling/Exp:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(*
paddingSAME*
strides

  
learnable_pooling/SqueezeSqueeze$learnable_pooling/depthwise:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(*
squeeze_dims
N
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7
MaximumMaximum"learnable_pooling/Squeeze:output:0Maximum/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(|
PCEN/Minimum/ReadVariableOpReadVariableOp$pcen_minimum_readvariableop_resource*
_output_shapes
:(*
dtype0S
PCEN/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
PCEN/MinimumMinimum#PCEN/Minimum/ReadVariableOp:value:0PCEN/Minimum/y:output:0*
T0*
_output_shapes
:(|
PCEN/Maximum/ReadVariableOpReadVariableOp$pcen_maximum_readvariableop_resource*
_output_shapes
:(*
dtype0S
PCEN/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?z
PCEN/MaximumMaximum#PCEN/Maximum/ReadVariableOp:value:0PCEN/Maximum/y:output:0*
T0*
_output_shapes
:(W
PCEN/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B : T
PCEN/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :¸
PCEN/GatherV2GatherV2Maximum:z:0PCEN/GatherV2/indices:output:0PCEN/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
%PCEN/EMA/clip_by_value/ReadVariableOpReadVariableOp.pcen_ema_clip_by_value_readvariableop_resource*
_output_shapes
:(*
dtype0e
 PCEN/EMA/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¨
PCEN/EMA/clip_by_value/MinimumMinimum-PCEN/EMA/clip_by_value/ReadVariableOp:value:0)PCEN/EMA/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:(]
PCEN/EMA/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
PCEN/EMA/clip_by_valueMaximum"PCEN/EMA/clip_by_value/Minimum:z:0!PCEN/EMA/clip_by_value/y:output:0*
T0*
_output_shapes
:(l
PCEN/EMA/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
PCEN/EMA/transpose	TransposeMaximum:z:0 PCEN/EMA/transpose/perm:output:0*
T0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿ(z
)PCEN/EMA/scan/TensorArrayV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   j
(PCEN/EMA/scan/TensorArrayV2/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :dé
PCEN/EMA/scan/TensorArrayV2TensorListReserve2PCEN/EMA/scan/TensorArrayV2/element_shape:output:01PCEN/EMA/scan/TensorArrayV2/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
CPCEN/EMA/scan/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   
5PCEN/EMA/scan/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorPCEN/EMA/transpose:y:0LPCEN/EMA/scan/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ|
+PCEN/EMA/scan/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   l
*PCEN/EMA/scan/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :dï
PCEN/EMA/scan/TensorArrayV2_1TensorListReserve4PCEN/EMA/scan/TensorArrayV2_1/element_shape:output:03PCEN/EMA/scan/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒU
PCEN/EMA/scan/ConstConst*
_output_shapes
: *
dtype0*
value	B : h
&PCEN/EMA/scan/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
value	B :db
 PCEN/EMA/scan/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
PCEN/EMA/scan/whileStatelessWhile)PCEN/EMA/scan/while/loop_counter:output:0/PCEN/EMA/scan/while/maximum_iterations:output:0PCEN/EMA/scan/Const:output:0PCEN/GatherV2:output:0&PCEN/EMA/scan/TensorArrayV2_1:handle:0EPCEN/EMA/scan/TensorArrayUnstack/TensorListFromTensor:output_handle:0PCEN/EMA/clip_by_value:z:0*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*7
_output_shapes%
#: : : :ÿÿÿÿÿÿÿÿÿ(: : :(* 
_read_only_resource_inputs
 *
_stateful_parallelism( **
body"R 
PCEN_EMA_scan_while_body_14977**
cond"R 
PCEN_EMA_scan_while_cond_14976*6
output_shapes%
#: : : :ÿÿÿÿÿÿÿÿÿ(: : :(
>PCEN/EMA/scan/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   
0PCEN/EMA/scan/TensorArrayV2Stack/TensorListStackTensorListStackPCEN/EMA/scan/while:output:4GPCEN/EMA/scan/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿ(*
element_dtype0*
num_elementsdn
PCEN/EMA/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¶
PCEN/EMA/transpose_1	Transpose9PCEN/EMA/scan/TensorArrayV2Stack/TensorListStack:tensor:0"PCEN/EMA/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(S
PCEN/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?g
PCEN/truedivRealDivPCEN/truediv/x:output:0PCEN/Maximum:z:0*
T0*
_output_shapes
:(O

PCEN/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ì¼+v
PCEN/addAddV2PCEN/add/x:output:0PCEN/EMA/transpose_1:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(e
PCEN/powPowPCEN/add:z:0PCEN/Minimum:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(j
PCEN/truediv_1RealDivMaximum:z:0PCEN/pow:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(x
PCEN/add_1/ReadVariableOpReadVariableOp"pcen_add_1_readvariableop_resource*
_output_shapes
:(*
dtype0

PCEN/add_1AddV2PCEN/truediv_1:z:0!PCEN/add_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(i

PCEN/pow_1PowPCEN/add_1:z:0PCEN/truediv:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(r
PCEN/ReadVariableOpReadVariableOp"pcen_add_1_readvariableop_resource*
_output_shapes
:(*
dtype0e

PCEN/pow_2PowPCEN/ReadVariableOp:value:0PCEN/truediv:z:0*
T0*
_output_shapes
:(e
PCEN/subSubPCEN/pow_1:z:0PCEN/pow_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(_
IdentityIdentityPCEN/sub:z:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(Û
NoOpNoOp&^PCEN/EMA/clip_by_value/ReadVariableOp^PCEN/Maximum/ReadVariableOp^PCEN/Minimum/ReadVariableOp^PCEN/ReadVariableOp^PCEN/add_1/ReadVariableOp/^learnable_pooling/clip_by_value/ReadVariableOp$^tfbanks_complex_conv/ReadVariableOp&^tfbanks_complex_conv/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ}: : : : : : 2N
%PCEN/EMA/clip_by_value/ReadVariableOp%PCEN/EMA/clip_by_value/ReadVariableOp2:
PCEN/Maximum/ReadVariableOpPCEN/Maximum/ReadVariableOp2:
PCEN/Minimum/ReadVariableOpPCEN/Minimum/ReadVariableOp2*
PCEN/ReadVariableOpPCEN/ReadVariableOp26
PCEN/add_1/ReadVariableOpPCEN/add_1/ReadVariableOp2`
.learnable_pooling/clip_by_value/ReadVariableOp.learnable_pooling/clip_by_value/ReadVariableOp2J
#tfbanks_complex_conv/ReadVariableOp#tfbanks_complex_conv/ReadVariableOp2N
%tfbanks_complex_conv/ReadVariableOp_1%tfbanks_complex_conv/ReadVariableOp_1:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
 
_user_specified_nameinputs
Ö
a
E__inference_sequential_layer_call_and_return_conditional_losses_13993

inputs
identityÒ
$global_max_pooling2d/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_13955ß
flatten/PartitionedCallPartitionedCall-global_max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_13963h
IdentityIdentity flatten/PartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd(:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(
 
_user_specified_nameinputs
¬
Õ
!__inference__traced_restore_15866
file_prefix@
.assignvariableop_audio_classifier_dense_kernel:<
.assignvariableop_1_audio_classifier_dense_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: +
assignvariableop_7_kernel:([
Aassignvariableop_8_audio_classifier_leaf_learnable_pooling_kernel:(A
3assignvariableop_9_audio_classifier_leaf_pcen_alpha:(B
4assignvariableop_10_audio_classifier_leaf_pcen_delta:(A
3assignvariableop_11_audio_classifier_leaf_pcen_root:(G
9assignvariableop_12_audio_classifier_leaf_pcen_ema_smooth:(#
assignvariableop_13_total: #
assignvariableop_14_count: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: J
8assignvariableop_17_adam_audio_classifier_dense_kernel_m:D
6assignvariableop_18_adam_audio_classifier_dense_bias_m:3
!assignvariableop_19_adam_kernel_m:(c
Iassignvariableop_20_adam_audio_classifier_leaf_learnable_pooling_kernel_m:(I
;assignvariableop_21_adam_audio_classifier_leaf_pcen_alpha_m:(I
;assignvariableop_22_adam_audio_classifier_leaf_pcen_delta_m:(H
:assignvariableop_23_adam_audio_classifier_leaf_pcen_root_m:(N
@assignvariableop_24_adam_audio_classifier_leaf_pcen_ema_smooth_m:(J
8assignvariableop_25_adam_audio_classifier_dense_kernel_v:D
6assignvariableop_26_adam_audio_classifier_dense_bias_v:3
!assignvariableop_27_adam_kernel_v:(c
Iassignvariableop_28_adam_audio_classifier_leaf_learnable_pooling_kernel_v:(I
;assignvariableop_29_adam_audio_classifier_leaf_pcen_alpha_v:(I
;assignvariableop_30_adam_audio_classifier_leaf_pcen_delta_v:(H
:assignvariableop_31_adam_audio_classifier_leaf_pcen_root_v:(N
@assignvariableop_32_adam_audio_classifier_leaf_pcen_ema_smooth_v:(
identity_34¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Ê
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*ð
valueæBã"B'_head/kernel/.ATTRIBUTES/VARIABLE_VALUEB%_head/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBC_head/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBA_head/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBC_head/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBA_head/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH´
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ë
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp.assignvariableop_audio_classifier_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp.assignvariableop_1_audio_classifier_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_8AssignVariableOpAassignvariableop_8_audio_classifier_leaf_learnable_pooling_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_9AssignVariableOp3assignvariableop_9_audio_classifier_leaf_pcen_alphaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_10AssignVariableOp4assignvariableop_10_audio_classifier_leaf_pcen_deltaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_11AssignVariableOp3assignvariableop_11_audio_classifier_leaf_pcen_rootIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_12AssignVariableOp9assignvariableop_12_audio_classifier_leaf_pcen_ema_smoothIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_17AssignVariableOp8assignvariableop_17_adam_audio_classifier_dense_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_18AssignVariableOp6assignvariableop_18_adam_audio_classifier_dense_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp!assignvariableop_19_adam_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:º
AssignVariableOp_20AssignVariableOpIassignvariableop_20_adam_audio_classifier_leaf_learnable_pooling_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_21AssignVariableOp;assignvariableop_21_adam_audio_classifier_leaf_pcen_alpha_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_22AssignVariableOp;assignvariableop_22_adam_audio_classifier_leaf_pcen_delta_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_23AssignVariableOp:assignvariableop_23_adam_audio_classifier_leaf_pcen_root_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_24AssignVariableOp@assignvariableop_24_adam_audio_classifier_leaf_pcen_ema_smooth_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_25AssignVariableOp8assignvariableop_25_adam_audio_classifier_dense_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_26AssignVariableOp6assignvariableop_26_adam_audio_classifier_dense_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp!assignvariableop_27_adam_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:º
AssignVariableOp_28AssignVariableOpIassignvariableop_28_adam_audio_classifier_leaf_learnable_pooling_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_29AssignVariableOp;assignvariableop_29_adam_audio_classifier_leaf_pcen_alpha_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_30AssignVariableOp;assignvariableop_30_adam_audio_classifier_leaf_pcen_delta_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_31AssignVariableOp:assignvariableop_31_adam_audio_classifier_leaf_pcen_root_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_32AssignVariableOp@assignvariableop_32_adam_audio_classifier_leaf_pcen_ema_smooth_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ¥
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_34IdentityIdentity_33:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_34Identity_34:output:0*W
_input_shapesF
D: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ü%
ú
#leaf_PCEN_EMA_scan_while_body_14456B
>leaf_pcen_ema_scan_while_leaf_pcen_ema_scan_while_loop_counterH
Dleaf_pcen_ema_scan_while_leaf_pcen_ema_scan_while_maximum_iterations(
$leaf_pcen_ema_scan_while_placeholder*
&leaf_pcen_ema_scan_while_placeholder_1*
&leaf_pcen_ema_scan_while_placeholder_2}
yleaf_pcen_ema_scan_while_tensorarrayv2read_tensorlistgetitem_leaf_pcen_ema_scan_tensorarrayunstack_tensorlistfromtensor_0>
:leaf_pcen_ema_scan_while_mul_leaf_pcen_ema_clip_by_value_0%
!leaf_pcen_ema_scan_while_identity'
#leaf_pcen_ema_scan_while_identity_1'
#leaf_pcen_ema_scan_while_identity_2'
#leaf_pcen_ema_scan_while_identity_3'
#leaf_pcen_ema_scan_while_identity_4{
wleaf_pcen_ema_scan_while_tensorarrayv2read_tensorlistgetitem_leaf_pcen_ema_scan_tensorarrayunstack_tensorlistfromtensor<
8leaf_pcen_ema_scan_while_mul_leaf_pcen_ema_clip_by_value
Jleaf/PCEN/EMA/scan/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   
<leaf/PCEN/EMA/scan/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemyleaf_pcen_ema_scan_while_tensorarrayv2read_tensorlistgetitem_leaf_pcen_ema_scan_tensorarrayunstack_tensorlistfromtensor_0$leaf_pcen_ema_scan_while_placeholderSleaf/PCEN/EMA/scan/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*
element_dtype0Ö
leaf/PCEN/EMA/scan/while/mulMul:leaf_pcen_ema_scan_while_mul_leaf_pcen_ema_clip_by_value_0Cleaf/PCEN/EMA/scan/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(c
leaf/PCEN/EMA/scan/while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?­
leaf/PCEN/EMA/scan/while/subSub'leaf/PCEN/EMA/scan/while/sub/x:output:0:leaf_pcen_ema_scan_while_mul_leaf_pcen_ema_clip_by_value_0*
T0*
_output_shapes
:(¡
leaf/PCEN/EMA/scan/while/mul_1Mul leaf/PCEN/EMA/scan/while/sub:z:0&leaf_pcen_ema_scan_while_placeholder_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
leaf/PCEN/EMA/scan/while/addAddV2 leaf/PCEN/EMA/scan/while/mul:z:0"leaf/PCEN/EMA/scan/while/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
=leaf/PCEN/EMA/scan/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem&leaf_pcen_ema_scan_while_placeholder_2$leaf_pcen_ema_scan_while_placeholder leaf/PCEN/EMA/scan/while/add:z:0*
_output_shapes
: *
element_dtype0:éèÒb
 leaf/PCEN/EMA/scan/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
leaf/PCEN/EMA/scan/while/add_1AddV2$leaf_pcen_ema_scan_while_placeholder)leaf/PCEN/EMA/scan/while/add_1/y:output:0*
T0*
_output_shapes
: b
 leaf/PCEN/EMA/scan/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :³
leaf/PCEN/EMA/scan/while/add_2AddV2>leaf_pcen_ema_scan_while_leaf_pcen_ema_scan_while_loop_counter)leaf/PCEN/EMA/scan/while/add_2/y:output:0*
T0*
_output_shapes
: r
!leaf/PCEN/EMA/scan/while/IdentityIdentity"leaf/PCEN/EMA/scan/while/add_2:z:0*
T0*
_output_shapes
: 
#leaf/PCEN/EMA/scan/while/Identity_1IdentityDleaf_pcen_ema_scan_while_leaf_pcen_ema_scan_while_maximum_iterations*
T0*
_output_shapes
: t
#leaf/PCEN/EMA/scan/while/Identity_2Identity"leaf/PCEN/EMA/scan/while/add_1:z:0*
T0*
_output_shapes
: 
#leaf/PCEN/EMA/scan/while/Identity_3Identity leaf/PCEN/EMA/scan/while/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
#leaf/PCEN/EMA/scan/while/Identity_4IdentityMleaf/PCEN/EMA/scan/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "O
!leaf_pcen_ema_scan_while_identity*leaf/PCEN/EMA/scan/while/Identity:output:0"S
#leaf_pcen_ema_scan_while_identity_1,leaf/PCEN/EMA/scan/while/Identity_1:output:0"S
#leaf_pcen_ema_scan_while_identity_2,leaf/PCEN/EMA/scan/while/Identity_2:output:0"S
#leaf_pcen_ema_scan_while_identity_3,leaf/PCEN/EMA/scan/while/Identity_3:output:0"S
#leaf_pcen_ema_scan_while_identity_4,leaf/PCEN/EMA/scan/while/Identity_4:output:0"v
8leaf_pcen_ema_scan_while_mul_leaf_pcen_ema_clip_by_value:leaf_pcen_ema_scan_while_mul_leaf_pcen_ema_clip_by_value_0"ô
wleaf_pcen_ema_scan_while_tensorarrayv2read_tensorlistgetitem_leaf_pcen_ema_scan_tensorarrayunstack_tensorlistfromtensoryleaf_pcen_ema_scan_while_tensorarrayv2read_tensorlistgetitem_leaf_pcen_ema_scan_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : :ÿÿÿÿÿÿÿÿÿ(: : :(: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:(
Ø	
Ä
0__inference_audio_classifier_layer_call_fn_14170
input_1
unknown:(#
	unknown_0:(
	unknown_1:(
	unknown_2:(
	unknown_3:(
	unknown_4:(
	unknown_5:
	unknown_6:
identity¢StatefulPartitionedCall²
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_audio_classifier_layer_call_and_return_conditional_losses_14130o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ}: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
!
_user_specified_name	input_1
ø
Ñ
PCEN_EMA_scan_while_cond_152078
4pcen_ema_scan_while_pcen_ema_scan_while_loop_counter>
:pcen_ema_scan_while_pcen_ema_scan_while_maximum_iterations#
pcen_ema_scan_while_placeholder%
!pcen_ema_scan_while_placeholder_1%
!pcen_ema_scan_while_placeholder_2O
Kpcen_ema_scan_while_pcen_ema_scan_while_cond_15207___redundant_placeholder0O
Kpcen_ema_scan_while_pcen_ema_scan_while_cond_15207___redundant_placeholder1 
pcen_ema_scan_while_identity
\
PCEN/EMA/scan/while/Less/yConst*
_output_shapes
: *
dtype0*
value	B :d
PCEN/EMA/scan/while/LessLesspcen_ema_scan_while_placeholder#PCEN/EMA/scan/while/Less/y:output:0*
T0*
_output_shapes
: µ
PCEN/EMA/scan/while/Less_1Less4pcen_ema_scan_while_pcen_ema_scan_while_loop_counter:pcen_ema_scan_while_pcen_ema_scan_while_maximum_iterations*
T0*
_output_shapes
: 
PCEN/EMA/scan/while/LogicalAnd
LogicalAndPCEN/EMA/scan/while/Less_1:z:0PCEN/EMA/scan/while/Less:z:0*
_output_shapes
: m
PCEN/EMA/scan/while/IdentityIdentity"PCEN/EMA/scan/while/LogicalAnd:z:0*
T0
*
_output_shapes
: "E
pcen_ema_scan_while_identity%PCEN/EMA/scan/while/Identity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : :ÿÿÿÿÿÿÿÿÿ(: ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:

_output_shapes
: :

_output_shapes
::

_output_shapes
:

a
J__inference_squared_modulus_layer_call_and_return_conditional_losses_15437
x
identityc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          i
	transpose	Transposextranspose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @`
powPowtranspose:y:0pow/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}b
 average_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :
average_pooling1d/ExpandDims
ExpandDimspow:z:0)average_pooling1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}Â
average_pooling1d/AvgPoolAvgPool%average_pooling1d/ExpandDims:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}*
ksize
*
paddingVALID*
strides

average_pooling1d/SqueezeSqueeze"average_pooling1d/AvgPool:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}*
squeeze_dims
J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @u
mulMulmul/x:output:0"average_pooling1d/Squeeze:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}e
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          s
transpose_1	Transposemul:z:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}(\
IdentityIdentitytranspose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ}P:O K
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}P

_user_specified_namex

M
1__inference_average_pooling1d_layer_call_fn_15627

inputs
identityÐ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_average_pooling1d_layer_call_and_return_conditional_losses_13474v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

a
E__inference_sequential_layer_call_and_return_conditional_losses_15293

inputs
identity{
*global_max_pooling2d/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
global_max_pooling2d/MaxMaxinputs3global_max_pooling2d/Max/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
flatten/ReshapeReshape!global_max_pooling2d/Max:output:0flatten/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentityflatten/Reshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd(:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(
 
_user_specified_nameinputs
ø
Ñ
PCEN_EMA_scan_while_cond_149768
4pcen_ema_scan_while_pcen_ema_scan_while_loop_counter>
:pcen_ema_scan_while_pcen_ema_scan_while_maximum_iterations#
pcen_ema_scan_while_placeholder%
!pcen_ema_scan_while_placeholder_1%
!pcen_ema_scan_while_placeholder_2O
Kpcen_ema_scan_while_pcen_ema_scan_while_cond_14976___redundant_placeholder0O
Kpcen_ema_scan_while_pcen_ema_scan_while_cond_14976___redundant_placeholder1 
pcen_ema_scan_while_identity
\
PCEN/EMA/scan/while/Less/yConst*
_output_shapes
: *
dtype0*
value	B :d
PCEN/EMA/scan/while/LessLesspcen_ema_scan_while_placeholder#PCEN/EMA/scan/while/Less/y:output:0*
T0*
_output_shapes
: µ
PCEN/EMA/scan/while/Less_1Less4pcen_ema_scan_while_pcen_ema_scan_while_loop_counter:pcen_ema_scan_while_pcen_ema_scan_while_maximum_iterations*
T0*
_output_shapes
: 
PCEN/EMA/scan/while/LogicalAnd
LogicalAndPCEN/EMA/scan/while/Less_1:z:0PCEN/EMA/scan/while/Less:z:0*
_output_shapes
: m
PCEN/EMA/scan/while/IdentityIdentity"PCEN/EMA/scan/while/LogicalAnd:z:0*
T0
*
_output_shapes
: "E
pcen_ema_scan_while_identity%PCEN/EMA/scan/while/Identity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : :ÿÿÿÿÿÿÿÿÿ(: ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(:

_output_shapes
: :

_output_shapes
::

_output_shapes
:
õ
£
?__inference_leaf_layer_call_and_return_conditional_losses_13845

inputs,
tfbanks_complex_conv_13825:(1
learnable_pooling_13829:(

pcen_13834:(

pcen_13836:(

pcen_13838:(

pcen_13840:(

identity_1¢PCEN/StatefulPartitionedCall¢)learnable_pooling/StatefulPartitionedCall¢,tfbanks_complex_conv/StatefulPartitionedCallh
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         þ
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*

begin_mask*
end_mask*
new_axis_mask
,tfbanks_complex_conv/StatefulPartitionedCallStatefulPartitionedCallstrided_slice:output:0tfbanks_complex_conv_13825*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}P*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_tfbanks_complex_conv_layer_call_and_return_conditional_losses_13577ü
squared_modulus/PartitionedCallPartitionedCall5tfbanks_complex_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_squared_modulus_layer_call_and_return_conditional_losses_13597
)learnable_pooling/StatefulPartitionedCallStatefulPartitionedCall(squared_modulus/PartitionedCall:output:0learnable_pooling_13829*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_learnable_pooling_layer_call_and_return_conditional_losses_13633N
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7
MaximumMaximum2learnable_pooling/StatefulPartitionedCall:output:0Maximum/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(
PCEN/StatefulPartitionedCallStatefulPartitionedCallMaximum:z:0
pcen_13834
pcen_13836
pcen_13838
pcen_13840*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_PCEN_layer_call_and_return_conditional_losses_13737q
IdentityIdentity%PCEN/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(f

Identity_1IdentityIdentity:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(À
NoOpNoOp^PCEN/StatefulPartitionedCall*^learnable_pooling/StatefulPartitionedCall-^tfbanks_complex_conv/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ}: : : : : : 2<
PCEN/StatefulPartitionedCallPCEN/StatefulPartitionedCall2V
)learnable_pooling/StatefulPartitionedCall)learnable_pooling/StatefulPartitionedCall2\
,tfbanks_complex_conv/StatefulPartitionedCall,tfbanks_complex_conv/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
 
_user_specified_nameinputs
ø
¤
?__inference_leaf_layer_call_and_return_conditional_losses_13930
input_1,
tfbanks_complex_conv_13910:(1
learnable_pooling_13914:(

pcen_13919:(

pcen_13921:(

pcen_13923:(

pcen_13925:(

identity_1¢PCEN/StatefulPartitionedCall¢)learnable_pooling/StatefulPartitionedCall¢,tfbanks_complex_conv/StatefulPartitionedCallh
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ÿ
strided_sliceStridedSliceinput_1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*

begin_mask*
end_mask*
new_axis_mask
,tfbanks_complex_conv/StatefulPartitionedCallStatefulPartitionedCallstrided_slice:output:0tfbanks_complex_conv_13910*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}P*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_tfbanks_complex_conv_layer_call_and_return_conditional_losses_13577ü
squared_modulus/PartitionedCallPartitionedCall5tfbanks_complex_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_squared_modulus_layer_call_and_return_conditional_losses_13597
)learnable_pooling/StatefulPartitionedCallStatefulPartitionedCall(squared_modulus/PartitionedCall:output:0learnable_pooling_13914*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_learnable_pooling_layer_call_and_return_conditional_losses_13633N
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7
MaximumMaximum2learnable_pooling/StatefulPartitionedCall:output:0Maximum/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(
PCEN/StatefulPartitionedCallStatefulPartitionedCallMaximum:z:0
pcen_13919
pcen_13921
pcen_13923
pcen_13925*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_PCEN_layer_call_and_return_conditional_losses_13737q
IdentityIdentity%PCEN/StatefulPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(f

Identity_1IdentityIdentity:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(À
NoOpNoOp^PCEN/StatefulPartitionedCall*^learnable_pooling/StatefulPartitionedCall-^tfbanks_complex_conv/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ}: : : : : : 2<
PCEN/StatefulPartitionedCallPCEN/StatefulPartitionedCall2V
)learnable_pooling/StatefulPartitionedCall)learnable_pooling/StatefulPartitionedCall2\
,tfbanks_complex_conv/StatefulPartitionedCall,tfbanks_complex_conv/StatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
!
_user_specified_name	input_1
ã
k
O__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_13955

inputs
identityf
Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      d
MaxMaxinputsMax/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
IdentityIdentityMax:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd(:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(
 
_user_specified_nameinputs
Õ	
Ã
0__inference_audio_classifier_layer_call_fn_14291

inputs
unknown:(#
	unknown_0:(
	unknown_1:(
	unknown_2:(
	unknown_3:(
	unknown_4:(
	unknown_5:
	unknown_6:
identity¢StatefulPartitionedCall±
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_audio_classifier_layer_call_and_return_conditional_losses_14130o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ}: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
 
_user_specified_nameinputs
Ç

1__inference_learnable_pooling_layer_call_fn_15444

inputs!
unknown:(
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_learnable_pooling_layer_call_and_return_conditional_losses_13633s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:ÿÿÿÿÿÿÿÿÿ}(: 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}(
 
_user_specified_nameinputs
ô

$__inference_leaf_layer_call_fn_13763
input_1
unknown:(#
	unknown_0:(
	unknown_1:(
	unknown_2:(
	unknown_3:(
	unknown_4:(
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_leaf_layer_call_and_return_conditional_losses_13748s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ}: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
!
_user_specified_name	input_1
éõ
Ú
K__inference_audio_classifier_layer_call_and_return_conditional_losses_14778

inputsC
1leaf_tfbanks_complex_conv_readvariableop_resource:(V
<leaf_learnable_pooling_clip_by_value_readvariableop_resource:(7
)leaf_pcen_minimum_readvariableop_resource:(7
)leaf_pcen_maximum_readvariableop_resource:(A
3leaf_pcen_ema_clip_by_value_readvariableop_resource:(5
'leaf_pcen_add_1_readvariableop_resource:(6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢*leaf/PCEN/EMA/clip_by_value/ReadVariableOp¢ leaf/PCEN/Maximum/ReadVariableOp¢ leaf/PCEN/Minimum/ReadVariableOp¢leaf/PCEN/ReadVariableOp¢leaf/PCEN/add_1/ReadVariableOp¢3leaf/learnable_pooling/clip_by_value/ReadVariableOp¢(leaf/tfbanks_complex_conv/ReadVariableOp¢*leaf/tfbanks_complex_conv/ReadVariableOp_1m
leaf/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            o
leaf/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            o
leaf/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
leaf/strided_sliceStridedSliceinputs!leaf/strided_slice/stack:output:0#leaf/strided_slice/stack_1:output:0#leaf/strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*

begin_mask*
end_mask*
new_axis_mask
(leaf/tfbanks_complex_conv/ReadVariableOpReadVariableOp1leaf_tfbanks_complex_conv_readvariableop_resource*
_output_shapes

:(*
dtype0~
-leaf/tfbanks_complex_conv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
/leaf/tfbanks_complex_conv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
/leaf/tfbanks_complex_conv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
'leaf/tfbanks_complex_conv/strided_sliceStridedSlice0leaf/tfbanks_complex_conv/ReadVariableOp:value:06leaf/tfbanks_complex_conv/strided_slice/stack:output:08leaf/tfbanks_complex_conv/strided_slice/stack_1:output:08leaf/tfbanks_complex_conv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:(*

begin_mask*
end_mask*
shrink_axis_maskv
1leaf/tfbanks_complex_conv/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÛI@Í
/leaf/tfbanks_complex_conv/clip_by_value/MinimumMinimum0leaf/tfbanks_complex_conv/strided_slice:output:0:leaf/tfbanks_complex_conv/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:(n
)leaf/tfbanks_complex_conv/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    À
'leaf/tfbanks_complex_conv/clip_by_valueMaximum3leaf/tfbanks_complex_conv/clip_by_value/Minimum:z:02leaf/tfbanks_complex_conv/clip_by_value/y:output:0*
T0*
_output_shapes
:(
*leaf/tfbanks_complex_conv/ReadVariableOp_1ReadVariableOp1leaf_tfbanks_complex_conv_readvariableop_resource*
_output_shapes

:(*
dtype0
/leaf/tfbanks_complex_conv/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       
1leaf/tfbanks_complex_conv/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
1leaf/tfbanks_complex_conv/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
)leaf/tfbanks_complex_conv/strided_slice_1StridedSlice2leaf/tfbanks_complex_conv/ReadVariableOp_1:value:08leaf/tfbanks_complex_conv/strided_slice_1/stack:output:0:leaf/tfbanks_complex_conv/strided_slice_1/stack_1:output:0:leaf/tfbanks_complex_conv/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:(*

begin_mask*
end_mask*
shrink_axis_maskx
3leaf/tfbanks_complex_conv/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ICÓ
1leaf/tfbanks_complex_conv/clip_by_value_1/MinimumMinimum2leaf/tfbanks_complex_conv/strided_slice_1:output:0<leaf/tfbanks_complex_conv/clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes
:(p
+leaf/tfbanks_complex_conv/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *Tã¿?Æ
)leaf/tfbanks_complex_conv/clip_by_value_1Maximum5leaf/tfbanks_complex_conv/clip_by_value_1/Minimum:z:04leaf/tfbanks_complex_conv/clip_by_value_1/y:output:0*
T0*
_output_shapes
:(Á
leaf/tfbanks_complex_conv/stackPack+leaf/tfbanks_complex_conv/clip_by_value:z:0-leaf/tfbanks_complex_conv/clip_by_value_1:z:0*
N*
T0*
_output_shapes

:(*

axisj
%leaf/tfbanks_complex_conv/range/startConst*
_output_shapes
: *
dtype0*
valueB
 *  HÃj
%leaf/tfbanks_complex_conv/range/limitConst*
_output_shapes
: *
dtype0*
valueB
 *  ICj
%leaf/tfbanks_complex_conv/range/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?á
leaf/tfbanks_complex_conv/rangeRange.leaf/tfbanks_complex_conv/range/start:output:0.leaf/tfbanks_complex_conv/range/limit:output:0.leaf/tfbanks_complex_conv/range/delta:output:0*

Tidx0*
_output_shapes	
:
/leaf/tfbanks_complex_conv/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        
1leaf/tfbanks_complex_conv/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
1leaf/tfbanks_complex_conv/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
)leaf/tfbanks_complex_conv/strided_slice_2StridedSlice(leaf/tfbanks_complex_conv/stack:output:08leaf/tfbanks_complex_conv/strided_slice_2/stack:output:0:leaf/tfbanks_complex_conv/strided_slice_2/stack_1:output:0:leaf/tfbanks_complex_conv/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:(*

begin_mask*
end_mask*
shrink_axis_mask
/leaf/tfbanks_complex_conv/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       
1leaf/tfbanks_complex_conv/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
1leaf/tfbanks_complex_conv/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
)leaf/tfbanks_complex_conv/strided_slice_3StridedSlice(leaf/tfbanks_complex_conv/stack:output:08leaf/tfbanks_complex_conv/strided_slice_3/stack:output:0:leaf/tfbanks_complex_conv/strided_slice_3/stack_1:output:0:leaf/tfbanks_complex_conv/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:(*

begin_mask*
end_mask*
shrink_axis_maske
 leaf/tfbanks_complex_conv/Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ÛÉ@r
leaf/tfbanks_complex_conv/SqrtSqrt)leaf/tfbanks_complex_conv/Sqrt/x:output:0*
T0*
_output_shapes
: ¡
leaf/tfbanks_complex_conv/mulMul"leaf/tfbanks_complex_conv/Sqrt:y:02leaf/tfbanks_complex_conv/strided_slice_3:output:0*
T0*
_output_shapes
:(h
#leaf/tfbanks_complex_conv/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¢
!leaf/tfbanks_complex_conv/truedivRealDiv,leaf/tfbanks_complex_conv/truediv/x:output:0!leaf/tfbanks_complex_conv/mul:z:0*
T0*
_output_shapes
:(d
leaf/tfbanks_complex_conv/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @§
leaf/tfbanks_complex_conv/powPow2leaf/tfbanks_complex_conv/strided_slice_3:output:0(leaf/tfbanks_complex_conv/pow/y:output:0*
T0*
_output_shapes
:(f
!leaf/tfbanks_complex_conv/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
leaf/tfbanks_complex_conv/mul_1Mul*leaf/tfbanks_complex_conv/mul_1/x:output:0!leaf/tfbanks_complex_conv/pow:z:0*
T0*
_output_shapes
:(j
%leaf/tfbanks_complex_conv/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¨
#leaf/tfbanks_complex_conv/truediv_1RealDiv.leaf/tfbanks_complex_conv/truediv_1/x:output:0#leaf/tfbanks_complex_conv/mul_1:z:0*
T0*
_output_shapes
:(f
!leaf/tfbanks_complex_conv/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @¢
leaf/tfbanks_complex_conv/pow_1Pow(leaf/tfbanks_complex_conv/range:output:0*leaf/tfbanks_complex_conv/pow_1/y:output:0*
T0*
_output_shapes	
:o
leaf/tfbanks_complex_conv/NegNeg#leaf/tfbanks_complex_conv/pow_1:z:0*
T0*
_output_shapes	
:
1leaf/tfbanks_complex_conv/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"(      Ä
+leaf/tfbanks_complex_conv/Tensordot/ReshapeReshape'leaf/tfbanks_complex_conv/truediv_1:z:0:leaf/tfbanks_complex_conv/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:(
3leaf/tfbanks_complex_conv/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"     Ã
-leaf/tfbanks_complex_conv/Tensordot/Reshape_1Reshape!leaf/tfbanks_complex_conv/Neg:y:0<leaf/tfbanks_complex_conv/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes
:	Ì
*leaf/tfbanks_complex_conv/Tensordot/MatMulMatMul4leaf/tfbanks_complex_conv/Tensordot/Reshape:output:06leaf/tfbanks_complex_conv/Tensordot/Reshape_1:output:0*
T0*
_output_shapes
:	(
leaf/tfbanks_complex_conv/ExpExp4leaf/tfbanks_complex_conv/Tensordot/MatMul:product:0*
T0*
_output_shapes
:	(
leaf/tfbanks_complex_conv/CastCast2leaf/tfbanks_complex_conv/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
:(
 leaf/tfbanks_complex_conv/Cast_1Cast(leaf/tfbanks_complex_conv/range:output:0*

DstT0*

SrcT0*
_output_shapes	
:
3leaf/tfbanks_complex_conv/Tensordot_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"(      Ã
-leaf/tfbanks_complex_conv/Tensordot_1/ReshapeReshape"leaf/tfbanks_complex_conv/Cast:y:0<leaf/tfbanks_complex_conv/Tensordot_1/Reshape/shape:output:0*
T0*
_output_shapes

:(
5leaf/tfbanks_complex_conv/Tensordot_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"     Ê
/leaf/tfbanks_complex_conv/Tensordot_1/Reshape_1Reshape$leaf/tfbanks_complex_conv/Cast_1:y:0>leaf/tfbanks_complex_conv/Tensordot_1/Reshape_1/shape:output:0*
T0*
_output_shapes
:	Ò
,leaf/tfbanks_complex_conv/Tensordot_1/MatMulMatMul6leaf/tfbanks_complex_conv/Tensordot_1/Reshape:output:08leaf/tfbanks_complex_conv/Tensordot_1/Reshape_1:output:0*
T0*
_output_shapes
:	(j
!leaf/tfbanks_complex_conv/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB J      ?´
leaf/tfbanks_complex_conv/mul_2Mul*leaf/tfbanks_complex_conv/mul_2/x:output:06leaf/tfbanks_complex_conv/Tensordot_1/MatMul:product:0*
T0*
_output_shapes
:	(u
leaf/tfbanks_complex_conv/Exp_1Exp#leaf/tfbanks_complex_conv/mul_2:z:0*
T0*
_output_shapes
:	(
 leaf/tfbanks_complex_conv/Cast_2Cast%leaf/tfbanks_complex_conv/truediv:z:0*

DstT0*

SrcT0*
_output_shapes
:(
/leaf/tfbanks_complex_conv/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        
1leaf/tfbanks_complex_conv/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
1leaf/tfbanks_complex_conv/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      þ
)leaf/tfbanks_complex_conv/strided_slice_4StridedSlice$leaf/tfbanks_complex_conv/Cast_2:y:08leaf/tfbanks_complex_conv/strided_slice_4/stack:output:0:leaf/tfbanks_complex_conv/strided_slice_4/stack_1:output:0:leaf/tfbanks_complex_conv/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:(*

begin_mask*
end_mask*
new_axis_mask
 leaf/tfbanks_complex_conv/Cast_3Cast!leaf/tfbanks_complex_conv/Exp:y:0*

DstT0*

SrcT0*
_output_shapes
:	(©
leaf/tfbanks_complex_conv/mul_3Mul2leaf/tfbanks_complex_conv/strided_slice_4:output:0#leaf/tfbanks_complex_conv/Exp_1:y:0*
T0*
_output_shapes
:	(
leaf/tfbanks_complex_conv/mul_4Mul#leaf/tfbanks_complex_conv/mul_3:z:0$leaf/tfbanks_complex_conv/Cast_3:y:0*
T0*
_output_shapes
:	(l
leaf/tfbanks_complex_conv/RealReal#leaf/tfbanks_complex_conv/mul_4:z:0*
_output_shapes
:	(l
leaf/tfbanks_complex_conv/ImagImag#leaf/tfbanks_complex_conv/mul_4:z:0*
_output_shapes
:	(¾
!leaf/tfbanks_complex_conv/stack_1Pack'leaf/tfbanks_complex_conv/Real:output:0'leaf/tfbanks_complex_conv/Imag:output:0*
N*
T0*#
_output_shapes
:(*

axisx
'leaf/tfbanks_complex_conv/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"P     ´
!leaf/tfbanks_complex_conv/ReshapeReshape*leaf/tfbanks_complex_conv/stack_1:output:00leaf/tfbanks_complex_conv/Reshape/shape:output:0*
T0*
_output_shapes
:	Py
(leaf/tfbanks_complex_conv/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       ¹
#leaf/tfbanks_complex_conv/transpose	Transpose*leaf/tfbanks_complex_conv/Reshape:output:01leaf/tfbanks_complex_conv/transpose/perm:output:0*
T0*
_output_shapes
:	Pj
(leaf/tfbanks_complex_conv/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :¼
$leaf/tfbanks_complex_conv/ExpandDims
ExpandDims'leaf/tfbanks_complex_conv/transpose:y:01leaf/tfbanks_complex_conv/ExpandDims/dim:output:0*
T0*#
_output_shapes
:Pz
/leaf/tfbanks_complex_conv/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿË
+leaf/tfbanks_complex_conv/conv1d/ExpandDims
ExpandDimsleaf/strided_slice:output:08leaf/tfbanks_complex_conv/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}s
1leaf/tfbanks_complex_conv/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Ø
-leaf/tfbanks_complex_conv/conv1d/ExpandDims_1
ExpandDims-leaf/tfbanks_complex_conv/ExpandDims:output:0:leaf/tfbanks_complex_conv/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:Pû
 leaf/tfbanks_complex_conv/conv1dConv2D4leaf/tfbanks_complex_conv/conv1d/ExpandDims:output:06leaf/tfbanks_complex_conv/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}P*
paddingSAME*
strides
µ
(leaf/tfbanks_complex_conv/conv1d/SqueezeSqueeze)leaf/tfbanks_complex_conv/conv1d:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}P*
squeeze_dims

ýÿÿÿÿÿÿÿÿx
#leaf/squared_modulus/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ã
leaf/squared_modulus/transpose	Transpose1leaf/tfbanks_complex_conv/conv1d/Squeeze:output:0,leaf/squared_modulus/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}_
leaf/squared_modulus/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
leaf/squared_modulus/powPow"leaf/squared_modulus/transpose:y:0#leaf/squared_modulus/pow/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}w
5leaf/squared_modulus/average_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Ø
1leaf/squared_modulus/average_pooling1d/ExpandDims
ExpandDimsleaf/squared_modulus/pow:z:0>leaf/squared_modulus/average_pooling1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿP}ì
.leaf/squared_modulus/average_pooling1d/AvgPoolAvgPool:leaf/squared_modulus/average_pooling1d/ExpandDims:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}*
ksize
*
paddingVALID*
strides
À
.leaf/squared_modulus/average_pooling1d/SqueezeSqueeze7leaf/squared_modulus/average_pooling1d/AvgPool:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}*
squeeze_dims
_
leaf/squared_modulus/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @´
leaf/squared_modulus/mulMul#leaf/squared_modulus/mul/x:output:07leaf/squared_modulus/average_pooling1d/Squeeze:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(}z
%leaf/squared_modulus/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ²
 leaf/squared_modulus/transpose_1	Transposeleaf/squared_modulus/mul:z:0.leaf/squared_modulus/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}(¸
3leaf/learnable_pooling/clip_by_value/ReadVariableOpReadVariableOp<leaf_learnable_pooling_clip_by_value_readvariableop_resource*&
_output_shapes
:(*
dtype0s
.leaf/learnable_pooling/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Þ
,leaf/learnable_pooling/clip_by_value/MinimumMinimum;leaf/learnable_pooling/clip_by_value/ReadVariableOp:value:07leaf/learnable_pooling/clip_by_value/Minimum/y:output:0*
T0*&
_output_shapes
:(k
&leaf/learnable_pooling/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *rn£;Ã
$leaf/learnable_pooling/clip_by_valueMaximum0leaf/learnable_pooling/clip_by_value/Minimum:z:0/leaf/learnable_pooling/clip_by_value/y:output:0*
T0*&
_output_shapes
:(g
"leaf/learnable_pooling/range/startConst*
_output_shapes
: *
dtype0*
valueB
 *    g
"leaf/learnable_pooling/range/limitConst*
_output_shapes
: *
dtype0*
valueB
 * ÈCg
"leaf/learnable_pooling/range/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Õ
leaf/learnable_pooling/rangeRange+leaf/learnable_pooling/range/start:output:0+leaf/learnable_pooling/range/limit:output:0+leaf/learnable_pooling/range/delta:output:0*

Tidx0*
_output_shapes	
:}
$leaf/learnable_pooling/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"           ±
leaf/learnable_pooling/ReshapeReshape%leaf/learnable_pooling/range:output:0-leaf/learnable_pooling/Reshape/shape:output:0*
T0*'
_output_shapes
:a
leaf/learnable_pooling/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HC£
leaf/learnable_pooling/subSub'leaf/learnable_pooling/Reshape:output:0%leaf/learnable_pooling/sub/y:output:0*
T0*'
_output_shapes
:a
leaf/learnable_pooling/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?£
leaf/learnable_pooling/mulMul(leaf/learnable_pooling/clip_by_value:z:0%leaf/learnable_pooling/mul/y:output:0*
T0*&
_output_shapes
:(c
leaf/learnable_pooling/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ÈC
leaf/learnable_pooling/mul_1Mulleaf/learnable_pooling/mul:z:0'leaf/learnable_pooling/mul_1/y:output:0*
T0*&
_output_shapes
:(
leaf/learnable_pooling/truedivRealDivleaf/learnable_pooling/sub:z:0 leaf/learnable_pooling/mul_1:z:0*
T0*'
_output_shapes
:(a
leaf/learnable_pooling/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
leaf/learnable_pooling/powPow"leaf/learnable_pooling/truediv:z:0%leaf/learnable_pooling/pow/y:output:0*
T0*'
_output_shapes
:(c
leaf/learnable_pooling/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ¿
leaf/learnable_pooling/mul_2Mul'leaf/learnable_pooling/mul_2/x:output:0leaf/learnable_pooling/pow:z:0*
T0*'
_output_shapes
:(u
leaf/learnable_pooling/ExpExp leaf/learnable_pooling/mul_2:z:0*
T0*'
_output_shapes
:(g
%leaf/learnable_pooling/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :À
!leaf/learnable_pooling/ExpandDims
ExpandDims$leaf/squared_modulus/transpose_1:y:0.leaf/learnable_pooling/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}(
&leaf/learnable_pooling/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"     (      
.leaf/learnable_pooling/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      é
 leaf/learnable_pooling/depthwiseDepthwiseConv2dNative*leaf/learnable_pooling/ExpandDims:output:0leaf/learnable_pooling/Exp:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(*
paddingSAME*
strides

  ¡
leaf/learnable_pooling/SqueezeSqueeze)leaf/learnable_pooling/depthwise:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(*
squeeze_dims
S
leaf/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7
leaf/MaximumMaximum'leaf/learnable_pooling/Squeeze:output:0leaf/Maximum/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(
 leaf/PCEN/Minimum/ReadVariableOpReadVariableOp)leaf_pcen_minimum_readvariableop_resource*
_output_shapes
:(*
dtype0X
leaf/PCEN/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
leaf/PCEN/MinimumMinimum(leaf/PCEN/Minimum/ReadVariableOp:value:0leaf/PCEN/Minimum/y:output:0*
T0*
_output_shapes
:(
 leaf/PCEN/Maximum/ReadVariableOpReadVariableOp)leaf_pcen_maximum_readvariableop_resource*
_output_shapes
:(*
dtype0X
leaf/PCEN/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
leaf/PCEN/MaximumMaximum(leaf/PCEN/Maximum/ReadVariableOp:value:0leaf/PCEN/Maximum/y:output:0*
T0*
_output_shapes
:(\
leaf/PCEN/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B : Y
leaf/PCEN/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :Ì
leaf/PCEN/GatherV2GatherV2leaf/Maximum:z:0#leaf/PCEN/GatherV2/indices:output:0 leaf/PCEN/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
*leaf/PCEN/EMA/clip_by_value/ReadVariableOpReadVariableOp3leaf_pcen_ema_clip_by_value_readvariableop_resource*
_output_shapes
:(*
dtype0j
%leaf/PCEN/EMA/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?·
#leaf/PCEN/EMA/clip_by_value/MinimumMinimum2leaf/PCEN/EMA/clip_by_value/ReadVariableOp:value:0.leaf/PCEN/EMA/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes
:(b
leaf/PCEN/EMA/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
leaf/PCEN/EMA/clip_by_valueMaximum'leaf/PCEN/EMA/clip_by_value/Minimum:z:0&leaf/PCEN/EMA/clip_by_value/y:output:0*
T0*
_output_shapes
:(q
leaf/PCEN/EMA/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
leaf/PCEN/EMA/transpose	Transposeleaf/Maximum:z:0%leaf/PCEN/EMA/transpose/perm:output:0*
T0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿ(
.leaf/PCEN/EMA/scan/TensorArrayV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   o
-leaf/PCEN/EMA/scan/TensorArrayV2/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :dø
 leaf/PCEN/EMA/scan/TensorArrayV2TensorListReserve7leaf/PCEN/EMA/scan/TensorArrayV2/element_shape:output:06leaf/PCEN/EMA/scan/TensorArrayV2/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Hleaf/PCEN/EMA/scan/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   
:leaf/PCEN/EMA/scan/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorleaf/PCEN/EMA/transpose:y:0Qleaf/PCEN/EMA/scan/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
0leaf/PCEN/EMA/scan/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   q
/leaf/PCEN/EMA/scan/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :dþ
"leaf/PCEN/EMA/scan/TensorArrayV2_1TensorListReserve9leaf/PCEN/EMA/scan/TensorArrayV2_1/element_shape:output:08leaf/PCEN/EMA/scan/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒZ
leaf/PCEN/EMA/scan/ConstConst*
_output_shapes
: *
dtype0*
value	B : m
+leaf/PCEN/EMA/scan/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
value	B :dg
%leaf/PCEN/EMA/scan/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ì
leaf/PCEN/EMA/scan/whileStatelessWhile.leaf/PCEN/EMA/scan/while/loop_counter:output:04leaf/PCEN/EMA/scan/while/maximum_iterations:output:0!leaf/PCEN/EMA/scan/Const:output:0leaf/PCEN/GatherV2:output:0+leaf/PCEN/EMA/scan/TensorArrayV2_1:handle:0Jleaf/PCEN/EMA/scan/TensorArrayUnstack/TensorListFromTensor:output_handle:0leaf/PCEN/EMA/clip_by_value:z:0*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*7
_output_shapes%
#: : : :ÿÿÿÿÿÿÿÿÿ(: : :(* 
_read_only_resource_inputs
 *
_stateful_parallelism( */
body'R%
#leaf_PCEN_EMA_scan_while_body_14699*/
cond'R%
#leaf_PCEN_EMA_scan_while_cond_14698*6
output_shapes%
#: : : :ÿÿÿÿÿÿÿÿÿ(: : :(
Cleaf/PCEN/EMA/scan/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ(   
5leaf/PCEN/EMA/scan/TensorArrayV2Stack/TensorListStackTensorListStack!leaf/PCEN/EMA/scan/while:output:4Lleaf/PCEN/EMA/scan/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:dÿÿÿÿÿÿÿÿÿ(*
element_dtype0*
num_elementsds
leaf/PCEN/EMA/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Å
leaf/PCEN/EMA/transpose_1	Transpose>leaf/PCEN/EMA/scan/TensorArrayV2Stack/TensorListStack:tensor:0'leaf/PCEN/EMA/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(X
leaf/PCEN/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?v
leaf/PCEN/truedivRealDivleaf/PCEN/truediv/x:output:0leaf/PCEN/Maximum:z:0*
T0*
_output_shapes
:(T
leaf/PCEN/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ì¼+
leaf/PCEN/addAddV2leaf/PCEN/add/x:output:0leaf/PCEN/EMA/transpose_1:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(t
leaf/PCEN/powPowleaf/PCEN/add:z:0leaf/PCEN/Minimum:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(y
leaf/PCEN/truediv_1RealDivleaf/Maximum:z:0leaf/PCEN/pow:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(
leaf/PCEN/add_1/ReadVariableOpReadVariableOp'leaf_pcen_add_1_readvariableop_resource*
_output_shapes
:(*
dtype0
leaf/PCEN/add_1AddV2leaf/PCEN/truediv_1:z:0&leaf/PCEN/add_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(x
leaf/PCEN/pow_1Powleaf/PCEN/add_1:z:0leaf/PCEN/truediv:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(|
leaf/PCEN/ReadVariableOpReadVariableOp'leaf_pcen_add_1_readvariableop_resource*
_output_shapes
:(*
dtype0t
leaf/PCEN/pow_2Pow leaf/PCEN/ReadVariableOp:value:0leaf/PCEN/truediv:z:0*
T0*
_output_shapes
:(t
leaf/PCEN/subSubleaf/PCEN/pow_1:z:0leaf/PCEN/pow_2:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(b
leaf/IdentityIdentityleaf/PCEN/sub:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(Y
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ

ExpandDims
ExpandDimsleaf/Identity:output:0ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(
5sequential/global_max_pooling2d/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ±
#sequential/global_max_pooling2d/MaxMaxExpandDims:output:0>sequential/global_max_pooling2d/Max/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¨
sequential/flatten/ReshapeReshape,sequential/global_max_pooling2d/Max:output:0!sequential/flatten/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense/MatMulMatMul#sequential/flatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp+^leaf/PCEN/EMA/clip_by_value/ReadVariableOp!^leaf/PCEN/Maximum/ReadVariableOp!^leaf/PCEN/Minimum/ReadVariableOp^leaf/PCEN/ReadVariableOp^leaf/PCEN/add_1/ReadVariableOp4^leaf/learnable_pooling/clip_by_value/ReadVariableOp)^leaf/tfbanks_complex_conv/ReadVariableOp+^leaf/tfbanks_complex_conv/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ}: : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2X
*leaf/PCEN/EMA/clip_by_value/ReadVariableOp*leaf/PCEN/EMA/clip_by_value/ReadVariableOp2D
 leaf/PCEN/Maximum/ReadVariableOp leaf/PCEN/Maximum/ReadVariableOp2D
 leaf/PCEN/Minimum/ReadVariableOp leaf/PCEN/Minimum/ReadVariableOp24
leaf/PCEN/ReadVariableOpleaf/PCEN/ReadVariableOp2@
leaf/PCEN/add_1/ReadVariableOpleaf/PCEN/add_1/ReadVariableOp2j
3leaf/learnable_pooling/clip_by_value/ReadVariableOp3leaf/learnable_pooling/clip_by_value/ReadVariableOp2T
(leaf/tfbanks_complex_conv/ReadVariableOp(leaf/tfbanks_complex_conv/ReadVariableOp2X
*leaf/tfbanks_complex_conv/ReadVariableOp_1*leaf/tfbanks_complex_conv/ReadVariableOp_1:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
 
_user_specified_nameinputs
õ
Ë
L__inference_learnable_pooling_layer_call_and_return_conditional_losses_13633

inputs?
%clip_by_value_readvariableop_resource:(
identity¢clip_by_value/ReadVariableOp
clip_by_value/ReadVariableOpReadVariableOp%clip_by_value_readvariableop_resource*&
_output_shapes
:(*
dtype0\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
clip_by_value/MinimumMinimum$clip_by_value/ReadVariableOp:value:0 clip_by_value/Minimum/y:output:0*
T0*&
_output_shapes
:(T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *rn£;~
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*&
_output_shapes
:(P
range/startConst*
_output_shapes
: *
dtype0*
valueB
 *    P
range/limitConst*
_output_shapes
: *
dtype0*
valueB
 * ÈCP
range/deltaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?y
rangeRangerange/start:output:0range/limit:output:0range/delta:output:0*

Tidx0*
_output_shapes	
:f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"           l
ReshapeReshaperange:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HC^
subSubReshape:output:0sub/y:output:0*
T0*'
_output_shapes
:J
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?^
mulMulclip_by_value:z:0mul/y:output:0*
T0*&
_output_shapes
:(L
mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ÈCX
mul_1Mulmul:z:0mul_1/y:output:0*
T0*&
_output_shapes
:(X
truedivRealDivsub:z:0	mul_1:z:0*
T0*'
_output_shapes
:(J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @Y
powPowtruediv:z:0pow/y:output:0*
T0*'
_output_shapes
:(L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ¿Y
mul_2Mulmul_2/x:output:0pow:z:0*
T0*'
_output_shapes
:(G
ExpExp	mul_2:z:0*
T0*'
_output_shapes
:(P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :t

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}(h
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"     (      h
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ¤
	depthwiseDepthwiseConv2dNativeExpandDims:output:0Exp:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(*
paddingSAME*
strides

  s
SqueezeSqueezedepthwise:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(*
squeeze_dims
c
IdentityIdentitySqueeze:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(e
NoOpNoOp^clip_by_value/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:ÿÿÿÿÿÿÿÿÿ}(: 2<
clip_by_value/ReadVariableOpclip_by_value/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}(
 
_user_specified_nameinputs
Ö
a
E__inference_sequential_layer_call_and_return_conditional_losses_13966

inputs
identityÒ
$global_max_pooling2d/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_13955ß
flatten/PartitionedCallPartitionedCall-global_max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_13963h
IdentityIdentity flatten/PartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿd(:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(
 
_user_specified_nameinputs
Ç

K__inference_audio_classifier_layer_call_and_return_conditional_losses_14220
input_1

leaf_14198:($

leaf_14200:(

leaf_14202:(

leaf_14204:(

leaf_14206:(

leaf_14208:(
dense_14214:
dense_14216:
identity¢dense/StatefulPartitionedCall¢leaf/StatefulPartitionedCall
leaf/StatefulPartitionedCallStatefulPartitionedCallinput_1
leaf_14198
leaf_14200
leaf_14202
leaf_14204
leaf_14206
leaf_14208*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_leaf_layer_call_and_return_conditional_losses_13845Y
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ

ExpandDims
ExpandDims%leaf/StatefulPartitionedCall:output:0ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(Ë
sequential/PartitionedCallPartitionedCallExpandDims:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_13993
dense/StatefulPartitionedCallStatefulPartitionedCall#sequential/PartitionedCall:output:0dense_14214dense_14216*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_14046u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense/StatefulPartitionedCall^leaf/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ}: : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2<
leaf/StatefulPartitionedCallleaf/StatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
!
_user_specified_name	input_1

¢
?__inference_leaf_layer_call_and_return_conditional_losses_13903
input_1,
tfbanks_complex_conv_13884:(1
learnable_pooling_13888:(

pcen_13893:(

pcen_13895:(

pcen_13897:(

pcen_13899:(
identity¢PCEN/StatefulPartitionedCall¢)learnable_pooling/StatefulPartitionedCall¢,tfbanks_complex_conv/StatefulPartitionedCallh
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         ÿ
strided_sliceStridedSliceinput_1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}*

begin_mask*
end_mask*
new_axis_mask
,tfbanks_complex_conv/StatefulPartitionedCallStatefulPartitionedCallstrided_slice:output:0tfbanks_complex_conv_13884*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}P*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_tfbanks_complex_conv_layer_call_and_return_conditional_losses_13577ü
squared_modulus/PartitionedCallPartitionedCall5tfbanks_complex_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_squared_modulus_layer_call_and_return_conditional_losses_13597
)learnable_pooling/StatefulPartitionedCallStatefulPartitionedCall(squared_modulus/PartitionedCall:output:0learnable_pooling_13888*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_learnable_pooling_layer_call_and_return_conditional_losses_13633N
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Å'7
MaximumMaximum2learnable_pooling/StatefulPartitionedCall:output:0Maximum/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(
PCEN/StatefulPartitionedCallStatefulPartitionedCallMaximum:z:0
pcen_13893
pcen_13895
pcen_13897
pcen_13899*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_PCEN_layer_call_and_return_conditional_losses_13737x
IdentityIdentity%PCEN/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(À
NoOpNoOp^PCEN/StatefulPartitionedCall*^learnable_pooling/StatefulPartitionedCall-^tfbanks_complex_conv/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ}: : : : : : 2<
PCEN/StatefulPartitionedCallPCEN/StatefulPartitionedCall2V
)learnable_pooling/StatefulPartitionedCall)learnable_pooling/StatefulPartitionedCall2\
,tfbanks_complex_conv/StatefulPartitionedCall,tfbanks_complex_conv/StatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
!
_user_specified_name	input_1
Ø	
Ä
0__inference_audio_classifier_layer_call_fn_14072
input_1
unknown:(#
	unknown_0:(
	unknown_1:(
	unknown_2:(
	unknown_3:(
	unknown_4:(
	unknown_5:
	unknown_6:
identity¢StatefulPartitionedCall²
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_audio_classifier_layer_call_and_return_conditional_losses_14053o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ}: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
!
_user_specified_name	input_1
 	
·
#__inference_signature_wrapper_14249
input_1
unknown:(#
	unknown_0:(
	unknown_1:(
	unknown_2:(
	unknown_3:(
	unknown_4:(
	unknown_5:
	unknown_6:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_13462o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ}: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
!
_user_specified_name	input_1

¿
$__inference_PCEN_layer_call_fn_15491

inputs
unknown:(
	unknown_0:(
	unknown_1:(
	unknown_2:(
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_PCEN_layer_call_and_return_conditional_losses_13737s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿd(: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd(
 
_user_specified_nameinputs"L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¬
serving_default
<
input_11
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ}<
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:¢Í

	_frontend
	_pool
	_head
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	
signatures
¢__call__
+£&call_and_return_all_conditional_losses
¤_default_save_signature"
_tf_keras_model
ë

_complex_conv
_activation
_pooling
_compress_fn
	variables
trainable_variables
regularization_losses
	keras_api
¥__call__
+¦&call_and_return_all_conditional_losses"
_tf_keras_model
Æ
layer-0
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
§__call__
+¨&call_and_return_all_conditional_losses"
_tf_keras_sequential
½

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
©__call__
+ª&call_and_return_all_conditional_losses"
_tf_keras_layer
ó
iter

beta_1

 beta_2
	!decay
"learning_ratemm#m$m%m&m'm(mvv#v$v%v&v'v (v¡"
	optimizer
X
#0
$1
%2
&3
'4
(5
6
7"
trackable_list_wrapper
X
#0
$1
%2
&3
'4
(5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
Î
)non_trainable_variables

*layers
+metrics
,layer_regularization_losses
-layer_metrics
	variables
trainable_variables
regularization_losses
¢__call__
¤_default_save_signature
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses"
_generic_user_object
-
«serving_default"
signature_map
À

#kernel
#_kernel
.	variables
/trainable_variables
0regularization_losses
1	keras_api
¬__call__
+­&call_and_return_all_conditional_losses"
_tf_keras_layer
²
	2_pool
3	variables
4trainable_variables
5regularization_losses
6	keras_api
®__call__
+¯&call_and_return_all_conditional_losses"
_tf_keras_layer
³

$kernel
7	variables
8trainable_variables
9regularization_losses
:	keras_api
°__call__
+±&call_and_return_all_conditional_losses"
_tf_keras_layer
Ð
	%alpha
	&delta
'root
;ema
<	variables
=trainable_variables
>regularization_losses
?	keras_api
²__call__
+³&call_and_return_all_conditional_losses"
_tf_keras_layer
J
#0
$1
%2
&3
'4
(5"
trackable_list_wrapper
J
#0
$1
%2
&3
'4
(5"
trackable_list_wrapper
 "
trackable_list_wrapper
°
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
	variables
trainable_variables
regularization_losses
¥__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses"
_generic_user_object
§
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
´__call__
+µ&call_and_return_all_conditional_losses"
_tf_keras_layer
§
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
¶__call__
+·&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
trainable_variables
regularization_losses
§__call__
+¨&call_and_return_all_conditional_losses
'¨"call_and_return_conditional_losses"
_generic_user_object
/:-2audio_classifier/dense/kernel
):'2audio_classifier/dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
	variables
trainable_variables
regularization_losses
©__call__
+ª&call_and_return_all_conditional_losses
'ª"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
:(2kernel
H:F(2.audio_classifier/leaf/learnable_pooling/kernel
.:,(2 audio_classifier/leaf/PCEN/alpha
.:,(2 audio_classifier/leaf/PCEN/delta
-:+(2audio_classifier/leaf/PCEN/root
3:1(2%audio_classifier/leaf/PCEN/EMA/smooth
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
#0"
trackable_list_wrapper
'
#0"
trackable_list_wrapper
 "
trackable_list_wrapper
°
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
.	variables
/trainable_variables
0regularization_losses
¬__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
§
^	variables
_trainable_variables
`regularization_losses
a	keras_api
¸__call__
+¹&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
3	variables
4trainable_variables
5regularization_losses
®__call__
+¯&call_and_return_all_conditional_losses
'¯"call_and_return_conditional_losses"
_generic_user_object
'
$0"
trackable_list_wrapper
'
$0"
trackable_list_wrapper
 "
trackable_list_wrapper
°
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
7	variables
8trainable_variables
9regularization_losses
°__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
Á

(smooth
(_weights
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
º__call__
+»&call_and_return_all_conditional_losses"
_tf_keras_layer
<
%0
&1
'2
(3"
trackable_list_wrapper
<
%0
&1
'2
(3"
trackable_list_wrapper
 "
trackable_list_wrapper
°
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
<	variables
=trainable_variables
>regularization_losses
²__call__
+³&call_and_return_all_conditional_losses
'³"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<

0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
´__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
¶__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Q
	total

count
	variables
	keras_api"
_tf_keras_metric
c

total

count

_fn_kwargs
	variables
	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
^	variables
_trainable_variables
`regularization_losses
¸__call__
+¹&call_and_return_all_conditional_losses
'¹"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
20"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
(0"
trackable_list_wrapper
'
(0"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
l	variables
mtrainable_variables
nregularization_losses
º__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
;0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
/
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
4:22$Adam/audio_classifier/dense/kernel/m
.:,2"Adam/audio_classifier/dense/bias/m
:(2Adam/kernel/m
M:K(25Adam/audio_classifier/leaf/learnable_pooling/kernel/m
3:1(2'Adam/audio_classifier/leaf/PCEN/alpha/m
3:1(2'Adam/audio_classifier/leaf/PCEN/delta/m
2:0(2&Adam/audio_classifier/leaf/PCEN/root/m
8:6(2,Adam/audio_classifier/leaf/PCEN/EMA/smooth/m
4:22$Adam/audio_classifier/dense/kernel/v
.:,2"Adam/audio_classifier/dense/bias/v
:(2Adam/kernel/v
M:K(25Adam/audio_classifier/leaf/learnable_pooling/kernel/v
3:1(2'Adam/audio_classifier/leaf/PCEN/alpha/v
3:1(2'Adam/audio_classifier/leaf/PCEN/delta/v
2:0(2&Adam/audio_classifier/leaf/PCEN/root/v
8:6(2,Adam/audio_classifier/leaf/PCEN/EMA/smooth/v
2ÿ
0__inference_audio_classifier_layer_call_fn_14072
0__inference_audio_classifier_layer_call_fn_14270
0__inference_audio_classifier_layer_call_fn_14291
0__inference_audio_classifier_layer_call_fn_14170´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
î2ë
K__inference_audio_classifier_layer_call_and_return_conditional_losses_14534
K__inference_audio_classifier_layer_call_and_return_conditional_losses_14778
K__inference_audio_classifier_layer_call_and_return_conditional_losses_14195
K__inference_audio_classifier_layer_call_and_return_conditional_losses_14220´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ËBÈ
 __inference__wrapped_model_13462input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
$__inference_leaf_layer_call_fn_13763
$__inference_leaf_layer_call_fn_14795
$__inference_leaf_layer_call_fn_14812
$__inference_leaf_layer_call_fn_13877´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¾2»
?__inference_leaf_layer_call_and_return_conditional_losses_15043
?__inference_leaf_layer_call_and_return_conditional_losses_15275
?__inference_leaf_layer_call_and_return_conditional_losses_13903
?__inference_leaf_layer_call_and_return_conditional_losses_13930´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ö2ó
*__inference_sequential_layer_call_fn_13969
*__inference_sequential_layer_call_fn_15280
*__inference_sequential_layer_call_fn_15285
*__inference_sequential_layer_call_fn_14001À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
â2ß
E__inference_sequential_layer_call_and_return_conditional_losses_15293
E__inference_sequential_layer_call_and_return_conditional_losses_15301
E__inference_sequential_layer_call_and_return_conditional_losses_14007
E__inference_sequential_layer_call_and_return_conditional_losses_14013À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ï2Ì
%__inference_dense_layer_call_fn_15310¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
@__inference_dense_layer_call_and_return_conditional_losses_15320¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÊBÇ
#__inference_signature_wrapper_14249input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Þ2Û
4__inference_tfbanks_complex_conv_layer_call_fn_15327¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ù2ö
O__inference_tfbanks_complex_conv_layer_call_and_return_conditional_losses_15416¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
/__inference_squared_modulus_layer_call_fn_15421
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
J__inference_squared_modulus_layer_call_and_return_conditional_losses_15437
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Û2Ø
1__inference_learnable_pooling_layer_call_fn_15444¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ö2ó
L__inference_learnable_pooling_layer_call_and_return_conditional_losses_15478¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Î2Ë
$__inference_PCEN_layer_call_fn_15491¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
é2æ
?__inference_PCEN_layer_call_and_return_conditional_losses_15589¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
4__inference_global_max_pooling2d_layer_call_fn_15594
4__inference_global_max_pooling2d_layer_call_fn_15599¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ê2Ç
O__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_15605
O__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_15611¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_flatten_layer_call_fn_15616¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_flatten_layer_call_and_return_conditional_losses_15622¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Û2Ø
1__inference_average_pooling1d_layer_call_fn_15627¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ö2ó
L__inference_average_pooling1d_layer_call_and_return_conditional_losses_15635¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¹2¶³
ª²¦
FullArgSpec.
args&#
jself
jinputs
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¹2¶³
ª²¦
FullArgSpec.
args&#
jself
jinputs
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ©
?__inference_PCEN_layer_call_and_return_conditional_losses_15589f%'(&3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿd(
ª ")¢&

0ÿÿÿÿÿÿÿÿÿd(
 
$__inference_PCEN_layer_call_fn_15491Y%'(&3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿd(
ª "ÿÿÿÿÿÿÿÿÿd(
 __inference__wrapped_model_13462r#$%'(&1¢.
'¢$
"
input_1ÿÿÿÿÿÿÿÿÿ}
ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ·
K__inference_audio_classifier_layer_call_and_return_conditional_losses_14195h#$%'(&5¢2
+¢(
"
input_1ÿÿÿÿÿÿÿÿÿ}
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ·
K__inference_audio_classifier_layer_call_and_return_conditional_losses_14220h#$%'(&5¢2
+¢(
"
input_1ÿÿÿÿÿÿÿÿÿ}
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¶
K__inference_audio_classifier_layer_call_and_return_conditional_losses_14534g#$%'(&4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ}
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¶
K__inference_audio_classifier_layer_call_and_return_conditional_losses_14778g#$%'(&4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ}
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
0__inference_audio_classifier_layer_call_fn_14072[#$%'(&5¢2
+¢(
"
input_1ÿÿÿÿÿÿÿÿÿ}
p 
ª "ÿÿÿÿÿÿÿÿÿ
0__inference_audio_classifier_layer_call_fn_14170[#$%'(&5¢2
+¢(
"
input_1ÿÿÿÿÿÿÿÿÿ}
p
ª "ÿÿÿÿÿÿÿÿÿ
0__inference_audio_classifier_layer_call_fn_14270Z#$%'(&4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ}
p 
ª "ÿÿÿÿÿÿÿÿÿ
0__inference_audio_classifier_layer_call_fn_14291Z#$%'(&4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ}
p
ª "ÿÿÿÿÿÿÿÿÿÕ
L__inference_average_pooling1d_layer_call_and_return_conditional_losses_15635E¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¬
1__inference_average_pooling1d_layer_call_fn_15627wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
@__inference_dense_layer_call_and_return_conditional_losses_15320\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 x
%__inference_dense_layer_call_fn_15310O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
B__inference_flatten_layer_call_and_return_conditional_losses_15622X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 v
'__inference_flatten_layer_call_fn_15616K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿØ
O__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_15605R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ³
O__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_15611`7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿd(
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¯
4__inference_global_max_pooling2d_layer_call_fn_15594wR¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
4__inference_global_max_pooling2d_layer_call_fn_15599S7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿd(
ª "ÿÿÿÿÿÿÿÿÿ­
?__inference_leaf_layer_call_and_return_conditional_losses_13903j#$%'(&5¢2
+¢(
"
input_1ÿÿÿÿÿÿÿÿÿ}
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿd(
 ­
?__inference_leaf_layer_call_and_return_conditional_losses_13930j#$%'(&5¢2
+¢(
"
input_1ÿÿÿÿÿÿÿÿÿ}
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿd(
 ¬
?__inference_leaf_layer_call_and_return_conditional_losses_15043i#$%'(&4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ}
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿd(
 ¬
?__inference_leaf_layer_call_and_return_conditional_losses_15275i#$%'(&4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ}
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿd(
 
$__inference_leaf_layer_call_fn_13763]#$%'(&5¢2
+¢(
"
input_1ÿÿÿÿÿÿÿÿÿ}
p 
ª "ÿÿÿÿÿÿÿÿÿd(
$__inference_leaf_layer_call_fn_13877]#$%'(&5¢2
+¢(
"
input_1ÿÿÿÿÿÿÿÿÿ}
p
ª "ÿÿÿÿÿÿÿÿÿd(
$__inference_leaf_layer_call_fn_14795\#$%'(&4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ}
p 
ª "ÿÿÿÿÿÿÿÿÿd(
$__inference_leaf_layer_call_fn_14812\#$%'(&4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ}
p
ª "ÿÿÿÿÿÿÿÿÿd(´
L__inference_learnable_pooling_layer_call_and_return_conditional_losses_15478d$4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ}(
ª ")¢&

0ÿÿÿÿÿÿÿÿÿd(
 
1__inference_learnable_pooling_layer_call_fn_15444W$4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ}(
ª "ÿÿÿÿÿÿÿÿÿd(Å
E__inference_sequential_layer_call_and_return_conditional_losses_14007|S¢P
I¢F
<9
global_max_pooling2d_inputÿÿÿÿÿÿÿÿÿd(
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Å
E__inference_sequential_layer_call_and_return_conditional_losses_14013|S¢P
I¢F
<9
global_max_pooling2d_inputÿÿÿÿÿÿÿÿÿd(
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ±
E__inference_sequential_layer_call_and_return_conditional_losses_15293h?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿd(
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ±
E__inference_sequential_layer_call_and_return_conditional_losses_15301h?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿd(
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_sequential_layer_call_fn_13969oS¢P
I¢F
<9
global_max_pooling2d_inputÿÿÿÿÿÿÿÿÿd(
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
*__inference_sequential_layer_call_fn_14001oS¢P
I¢F
<9
global_max_pooling2d_inputÿÿÿÿÿÿÿÿÿd(
p

 
ª "ÿÿÿÿÿÿÿÿÿ
*__inference_sequential_layer_call_fn_15280[?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿd(
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
*__inference_sequential_layer_call_fn_15285[?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿd(
p

 
ª "ÿÿÿÿÿÿÿÿÿ¤
#__inference_signature_wrapper_14249}#$%'(&<¢9
¢ 
2ª/
-
input_1"
input_1ÿÿÿÿÿÿÿÿÿ}"3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ«
J__inference_squared_modulus_layer_call_and_return_conditional_losses_15437]/¢,
%¢"
 
xÿÿÿÿÿÿÿÿÿ}P
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ}(
 
/__inference_squared_modulus_layer_call_fn_15421P/¢,
%¢"
 
xÿÿÿÿÿÿÿÿÿ}P
ª "ÿÿÿÿÿÿÿÿÿ}(¸
O__inference_tfbanks_complex_conv_layer_call_and_return_conditional_losses_15416e#4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ}
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ}P
 
4__inference_tfbanks_complex_conv_layer_call_fn_15327X#4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ}
ª "ÿÿÿÿÿÿÿÿÿ}P