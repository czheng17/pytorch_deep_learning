import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(1)

TIME_STEP = 10
INPUT_SIZE = 1
LR = 0.02

# # show data
# steps = np.linspace(0, np.pi*2, 100, dtype=np.float32)
# x_np = np.sin(steps)    # float32 for converting torch FloatTensor
# y_np = np.cos(steps)
# plt.plot(steps, y_np, 'r-', label='target (cos)')
# plt.plot(steps, x_np, 'b-', label='input (sin)')
# plt.legend(loc='best')
# plt.show()

class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn = torch.nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
        )
        self.out = torch.nn.Linear(
            in_features=32,
            out_features=1,
        )

    def forward(self, x,h_state):
        x_out, h_state = self.rnn(x)

        outs = []

        for time_step in range(x_out.size(1)):
            outs.append(self.out(x_out[ : , time_step , : ]))
        return torch.stack(outs,dim=1), h_state

rnn = RNN()

'''
print rnn:
RNN (
  (rnn): RNN(1, 32, batch_first=True)
  (out): Linear (32 -> 1)
)

'''

loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(),lr=LR)

h_state = None

plt.figure(1, figsize=(12, 5))
plt.ion() # continuously plot

for step in range(60):
    start,end = step*np.pi, (step+1)*np.pi

    steps = np.linspace(start,end,TIME_STEP,dtype=np.float32)
    x_np = np.sin(steps)
    y_np = np.cos(steps)

    x = Variable(torch.from_numpy(x_np[np.newaxis, : ,np.newaxis])) # shape (batch, time_step, input_size)
    y = Variable(torch.from_numpy(y_np[np.newaxis, : ,np.newaxis]))

    prediction, h_state = rnn(x,h_state)

    h_state = Variable(h_state.data)

    loss = loss_func(prediction,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.draw()
    plt.pause(0.05)

plt.ioff()
plt.show()


# np.newaxis:
# b = np.array([1,2,3,4,5,6])
# b[np.newaxis] ---> array([[1,2,3,4,5,6]])


# >>> start, end = 1*np.pi, 2*np.pi
# >>> start
# 3.141592653589793
# >>> end
# 6.283185307179586
# >>> steps = np.linspace(start,end,10,dtype=np.float32)
# >>> steps
# array([ 3.14159274,  3.49065852,  3.8397243 ,  4.18879032,  4.5378561 ,
#         4.88692188,  5.23598766,  5.58505344,  5.93411922,  6.28318548], dtype=float32)
#
# >>> x_np=np.sin(steps)
# >>> x_np
# array([ -8.74227766e-08,  -3.42020154e-01,  -6.42787576e-01,
#         -8.66025448e-01,  -9.84807789e-01,  -9.84807730e-01,
#         -8.66025448e-01,  -6.42787755e-01,  -3.42020363e-01,
#          1.74845553e-07], dtype=float32)
# >>> res = x_np[np.newaxis, : , np.newaxis]
# >>> res
# array([[[ -8.74227766e-08],
#         [ -3.42020154e-01],
#         [ -6.42787576e-01],
#         [ -8.66025448e-01],
#         [ -9.84807789e-01],
#         [ -9.84807730e-01],
#         [ -8.66025448e-01],
#         [ -6.42787755e-01],
#         [ -3.42020363e-01],
#         [  1.74845553e-07]]], dtype=float32)
