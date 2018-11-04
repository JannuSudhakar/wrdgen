import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from matplotlib import pyplot as plt
import sys
import time

def read_words(filename = "words.txt"):
    f = open(filename)
    words = f.read().split()
    f.close()
    words = [word.split('_')[0].lower() for word in words]
    return words

def create_letter_list(word_list):
    ret = ["strt","end"]
    for word in word_list:
        for letter in word:
            if(not letter in ret):
                ret.append(letter)
    return ret

class wrdgen(nn.Module):
    def __init__(self,letter_list):
        super(wrdgen,self).__init__()
        self.letter_list = letter_list

        self.statevectform1 = nn.Linear(750+len(self.letter_list),250)
        self.mem1 = nn.Linear(750+len(self.letter_list),500)
        self.switch1 = nn.Linear(750+len(self.letter_list),500)
        self.statevectform2 = nn.Linear(750,250)
        self.mem2 = nn.Linear(750,500)
        self.switch2 = nn.Linear(750,500)
        self.statevectform3 = nn.Linear(750,250)
        self.mem3 = nn.Linear(750,500)
        self.switch3 = nn.Linear(750,500)
        self.statevectform4 = nn.Linear(750,250)
        self.mem4 = nn.Linear(750,500)
        self.switch4 = nn.Linear(750,500)
        self.statevectform5 = nn.Linear(750,250)
        self.mem5 = nn.Linear(750,500)
        self.switch5 = nn.Linear(750,500)
        self.statevectform6 = nn.Linear(750,250)
        self.mem6 = nn.Linear(750,500)
        self.switch6 = nn.Linear(750,500)

        self.predictlayer1 = nn.Linear(750,450)
        self.predictlayer2 = nn.Linear(450,300)
        self.predictlayer3 = nn.Linear(300,200)
        self.predictlayer4 = nn.Linear(200,100)
        self.predictlayer5 = nn.Linear(100,len(self.letter_list))

        self.TH = nn.Tanh()
        self.SM = nn.Softmax(dim = 1)

    def one_step(self,x,m,letter):
        k = torch.zeros(1,len(self.letter_list))
        k[0,self.letter_list.index(letter)] = 1

        x = torch.cat((x,m,k),1)
        s = self.switch1(x).sigmoid()
        m = m*s + self.TH(self.mem1(x))*(1-s)
        x = self.TH(self.statevectform1(x))

        x = torch.cat((x,m),1)
        s = self.switch2(x).sigmoid()
        m = m*s + self.TH(self.mem2(x))*(1-s)
        x = self.TH(self.statevectform2(x))

        x = torch.cat((x,m),1)
        s = self.switch3(x).sigmoid()
        m = m*s + self.TH(self.mem3(x))*(1-s)
        x = self.TH(self.statevectform3(x))

        x = torch.cat((x,m),1)
        s = self.switch4(x).sigmoid()
        m = m*s + self.TH(self.mem4(x))*(1-s)
        x = self.TH(self.statevectform4(x))

        x = torch.cat((x,m),1)
        s = self.switch5(x).sigmoid()
        m = m*s + self.TH(self.mem5(x))*(1-s)
        x = self.TH(self.statevectform5(x))

        x = torch.cat((x,m),1)
        s = self.switch6(x).sigmoid()
        m = m*s + self.TH(self.mem6(x))*(1-s)
        x = self.TH(self.statevectform6(x))

        return x,m

    def predict(self,x,m):
        y = torch.cat((x,m),1)
        y = self.TH(self.predictlayer1(y))
        y = self.TH(self.predictlayer2(y))
        y = self.TH(self.predictlayer3(y))
        y = self.TH(self.predictlayer4(y))
        y = self.SM(self.predictlayer5(y))
        return y

    def forward(self,word):
        #this function doesn't work
        x = torch.zeros(1,500)
        x = self.one_step(x,'strt')
        for l in word:
            x = self.one_step(x,l)
        x = self.predict(x)
        return x

    def calculate_loss(self,h,y,criterion):
        yvec = torch.zeros(1,len(self.letter_list))
        yvec[0,self.letter_list.index(y)] = 1
        return criterion(h,yvec)

    def learn(self,word,criterion,optimizer):
        ret = 0
        x = torch.zeros(1,250)
        m = torch.zeros(1,500)
        x,m = self.one_step(x,m,'strt')
        h = self.predict(x,m)
        for i in range(len(word)):
            yvec = torch.zeros(1,len(self.letter_list))
            yvec[0,self.letter_list.index(word[i])] = 1
            loss = criterion(h,yvec)
            ret += loss.item()
            loss.backward(retain_graph = True)
            x,m = self.one_step(x,m,word[i])
            h = self.predict(x,m)
        yvec = torch.zeros(1,len(self.letter_list))
        yvec[0,self.letter_list.index('end')] = 1
        loss = criterion(h,yvec)
        ret += loss.item()
        loss.backward(retain_graph = True)
        optimizer.step()
        optimizer.zero_grad()
        ret = ret/(len(word) + 1)
        return ret

    def babble(self):
        wrd = ""
        x = torch.zeros(1,250)
        m = torch.zeros(1,500)
        x,m = self.one_step(x,m,'strt')
        y = self.predict(x,m).view(-1).detach().numpy()
        y = self.letter_list[np.random.choice(y.shape[0],p = y)]
        while (y != 'end'):
            wrd += y
            x,m = self.one_step(x,m,y)
            y = self.predict(x,m).view(-1).detach().numpy()
            y = self.letter_list[np.random.choice(y.shape[0],p = y)]
        return wrd

if (__name__ == "__main__"):
    words = read_words()
    letter_list = create_letter_list(words)
    model = wrdgen(letter_list)

    if(len(sys.argv) == 2):
        if(sys.argv[1] == 'g'):
            model.load_state_dict(torch.load("wrdgen.pt"))
            for i in range(100):
                print(model.babble())
            exit(0)
        if(sys.argv[1] == 'c'):
            model.load_state_dict(torch.load("wrdgen.pt"))

        if(sys.argv[1] == 't'):
            no_epochs = 100000
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(),lr = 0.0001,betas = (0.9,0.99))
            loss_mean = 0

            t1 = time.time()
            for epoch in range(no_epochs):
                wrd = words[np.random.randint(len(words))]
                """pos = np.random.randint(len(wrd))
                x = wrd[0:pos]
                if(pos == (len(wrd)-1)):
                    y = 'end'
                else: y = wrd[pos + 1]

                h = model(x)
                loss = model.calculate_loss(h,y,criterion)
                loss.backward()
                optimizer.step()
                """
                loss = model.learn(wrd,criterion,optimizer)
                loss_mean  = 0.9*loss_mean + 0.1*loss
                print("(epoch %d/time: %f)loss: %f; loss-mean: %f ----- %r"%(epoch,time.time()-t1,loss,loss_mean,wrd))

                optimizer.zero_grad()

                if((epoch+1)%100 == 0):
                    torch.save(model.state_dict(),"wrdgen.pt")

            t2 = time.time()
            print("runtime: ",t2-t1)

            for i in range(100):
                print(model.babble())

            torch.save(model.state_dict(),"wrdgen.pt")
