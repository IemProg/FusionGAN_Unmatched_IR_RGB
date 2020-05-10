class GeneratorTwo(nn.Module):
    def __init__(self, block):
        super(GeneratorTwo, self).__init__()
        
        self.conv1_x = nn.Sequential(                   #Down-sampling component
            nn.Conv2d(3, 16, (5, 5), stride  = 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(3, 16, (5, 5), stride = 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(3, 16, (3, 3), stride = 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(3, 16, (3, 3), stride = 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(3, 16, (1, 1), stride = 1),
            nn.Tanh()
        )

        #TO-DO List: 1\ check conv3_x:upsampling   2\ check parameters of layers        
    def forward(self, x, y):
        
        x = self.conv1_x(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        
        y = self.conv1_y(y)
        y = self.conv2_y(y)
        y = self.conv3_y(y)
        y = self.block1_y(y)
        y = self.block2_y(y)
        
#         if torch.cuda.is_available():
#             concat_result = torch.cuda.FloatTensor([x.shape[0], x.shape[1] * 2, x.shape[2], x.shape[3]]).fill_(0)
#         else:
        concat_result = torch.zeros([x.shape[0], x.shape[1] * 2, x.shape[2], x.shape[3]], dtype=x.dtype)
#         print(x.shape, y.shape, concat_result.shape)
        for i in range(batch_size):
            for j in range(x.shape[1]):
                concat_result[i][j] = x[i][j]
                concat_result[i][j + x.shape[1]] = y[i][j]
        if torch.cuda.is_available():
            concat_result = concat_result.cuda()
        concat_result = self.block1_concat(concat_result)
        concat_result = self.block2_concat(concat_result)
        
        upsampled_1 = self.upsample1(concat_result)
        upsampled_2 = self.upsample2(upsampled_1)
#         print(upsample2.shape)
        return upsampled_2