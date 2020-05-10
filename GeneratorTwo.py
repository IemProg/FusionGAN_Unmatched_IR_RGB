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
            nn.LeakyReLU(inplace=True)
        )

         self.conv2_x = nn.Sequential(    
            nn.Conv2d(3, 16, (1, 1), stride = 1),
            nn.Tanh()
        )

        #TO-DO List: 1\ check conv3_x:upsampling   2\ check parameters of layers        
    def forward(self, x):
        
        x = self.conv1_x(x)
        x = self.conv2_x(x)
        return x