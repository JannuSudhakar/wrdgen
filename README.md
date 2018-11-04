#What this code does
===================
This is code built on pytorch that trains a recurrent neural network
to output random english words (which are not actual english words).

The words look like well formed english words.

#Usage
============

The script can be interacted with from the commandline as follows:

    python wrdgen.py [flag]

[flag] can be the following:
* t: Trains the network and stores the trained weights at periodic
intervals in wrdgen.pt. The words used asa reference for training are
those in the words.txt file. The file as it stands contains words
extracted from the wordnet database. But it may be any file formatted
such that each line stands for a word. The network automatically
adapts its size to match the number of letters it was able to find in
the file (the letters are all converted to lowercase, if the file
contains words with special characters, the network will learn to
generate words with those special characters). The training is done
for 100000 epochs and the network is saved every 100 epochs. you can
manually stop the training after as many iterations as you like.
If the model does finish training for 100000 epochs it then outputs
100 random words which are proof of what it has managed to learn.
One needs around 60000 epochs for it to start outputing good sounding
words.
* c: Contiinues training the network stored in wrdgen.pt. Also the
file words.txt which is used to continue the training is expected to
be the same as the one with which the network was trained. If not,
bad things will happen.
* g: The script uses the file wrdgen.pt to now output 100 new words
previously unknown to english-kind. *The file words.txt is also used
and should be the same as during training. I might fix this later*
An example run is included at the end of this file.

#Architecture
=============
A simplified gated unit is used:

```s = M~s~*m + X~s~*x
m = s.m + (1-s).(M~m~*m + X~m~*x)
x = M~x~*m + X~x~*x
```

###Notation:
* All variables denoted by capital letters with subscripts are
parameter matrices of appropriate sizes.
* s is the vector used for gating the memory units.
* m is the part of the state vector that is designated as memory.
* x is the remaining part of the state vector which may change
in a more volatile manner.
* '.' denotes the elementwise multiplication operation
* '\*' denotes the matrix multiplication operation.
===================================================

###The GU:
Thus one gated unit (which we will abbreviate GU) operation on
a state vector consisting of (x~t-1~,m~t-1~) in this architecture
produces (x~t~,m~t~). Note that the sizes of the input state
vectors and the output state vectors need not be the same.

    (x~t-1~,m~t-1~) -GU-> (x~t~,m~t~)
==================================================

###The PRD:
Given a state vector so far the network outputs a prediction for
the next letter using a 6-layered fully connected network that
takes into account both x and m, with the final layer being a
softmax layer. The output is finally a vector that has the
probabilities of all the letters.

Let us call this fully connected network that does the predictions
as PRD.
    (x~n~,m~n~) -PRD-> PD~n~
which means the n^th^ state vector when input into the PRD outputs
the probability distribution over all possible letters that can be
the n^th^ letter in the word.
==================================================

###The OU:
When a new letter is given to the network, it updates its state
vector as follows:

1.The letter is first converted to a onehot vector and concatenated
with the x of the state vector:
    (x~0~,m~0~) = (x + one_hot(incoming_letter),m)
1.Then the state vector is passed through a number of gated units
(the number was 6 when this was written):
    (x~0~,m~0~) -GU-> (x~1~,m~1~) -GU-> (x~2,m~2) ...-GU-> (x~6~,m~6)

Let us call the above two operations together as one
operational-update (abbreviated OU), keep in mind that to perform
an OU we also need to know the next letter of the word.

    (x~n-1~,m~n-1~) -OU^incoming_letter^-> (x~n~,m~n~)
====================================================

###Train time:
At train time we initialize the state vector with zeros and run
one OU with the 'strt' character, which marks the beginning of the
word.
    (x~0~,m~0~) -OU^strt^-> (x~1~,m~1~)
At this point we have not yet fed in any information regarding the
word being trained on, the state vector now when passed through the
prediction network, should give us a probability distribution over
all possible letters that can be the first letter of this word.
    (x~1~,m~1~) -PRD-> PD~1~
A cross-entropy loss is now taken betwee PD~1~ and
one_hot(actual_first_letter), which is used for backpropagation.
    loss += CrossEntropy(PD~1~,one_hot(actual_first_letter))

The next OU is performed with the actual first letter of the word(l1)
PD~2~ is calculated for backpropagation purposes.
    (x~1~,m~1~) -OU^l1^-> (x~2~,m~2~)
    (x~2~,m~2~) -PRD-> PD~2~
    loss += CrossEntropy(PD~2~,one_hot(l2))

This set of operations is performed recursively until we get to the
end of the word marked by the 'end' character, which the network is
expected to predict (which is to say we backpropagate over the loss
incurred in predicting the 'end' character as well).

Finally we can backpropagate over the loss and make updates based on
the gradients obtained.

At periodic intervals the weights so obtained are promptly stored in
the 'wrdgen.pt' file in the folder where the training is happening.

When trained with the 'c' option (c stands for continue) the script
looks for 'wrdgen.pt' and loads the weights from there and further
trains those weights.
=====================================================

###Run time:
When run with the 'g' option (g stands for generate words) the script
looks for the 'wrdgen.pt' file and loads weights from there.

It then initializes (x~0~,m~0~) to zero vectors.
Then the following operations are run with l~0~ = 'strt' until
ln == 'end'
    (x~n-1~,m~n-1~) -OU^l(n-1)^-> (x~n~,m~n~)
    (x~n~,m~n~) -PRD-> PD~n~
    ln = pick(PD~n~)
    n++
the function pick(PD) picks a random letter with each letter's
probability of being picked being specified by the probability
distribution PD.

We then simply output the word consisting of all the letters that
were picked in order.
=======================================================

#Acknowledgements:
* Wordnet - As the time of writing of this file the file words.txt
was compiled using the word database of Wordnet.
  -Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010.
* Pytorch - The python library that implements backpropagation in
convenient ways - [Pytorch.org](https://pytorch.org/)
* Python - because if the python god that supports the entire land
of mortals on its vast and beautiful body didn't exist, we would all
be really good at swimming, some people say for the better, some say
otherwise. [Python.org](https://www.python.org/)
======================================================

#Example Run:
For example This was one run of the network generating random words:
```
$python wrdgen.py g
deprir
levananditis
oan
pliniosisly
graseous
graphaline
jyridae
unterined
recentic
prepargy
dromi
wrindy
dimclice
claastate
gurtible
sopguh
huuhbriver
morfyn
artodwove
bodatcic
serton
scincy
stligic
futtsian
sladendly
nudbrond
cranack
savifla
firt
agior
grymo
sepifeal
weaf
rosbin
norethere
odeded
concantally
paton
ala
epanium
bagors
orshuistzer
crucolally
upn
worce
vilisty
pedicepte
blpidius
dractinese
anhyquinus
stripied
howacteroe
stuzent
feacced
preatura
subaryis
sererious
retifemias
incitanum
nynse
unal
looth
milyyran
housubasectoy
icenatia
tountolus
pelogobe
reafnoe
nonfuter
hagomatict
nancy-aude
morname
thulilly
pirtowt
clugent
suppatipeae
thlytifieriss
ever-orate
medition
suppe
paymiker
cominestly
pucototice
igeonorist
coveodian
tirtara
bvoletificis
sourative
nonoman
terveegia
frith
ovysosis
henolad
monosuropheral
inean
rinturent
distromycled
canajus
britidiziao
nolic
$
```
