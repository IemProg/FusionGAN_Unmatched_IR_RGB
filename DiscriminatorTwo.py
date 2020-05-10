class DiscriminatorTwo(nn.Module):
    def __init__(self, block):
        super(DiscriminatorTwo, self).__init__()
        
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
            #TODO: reshape layer
        )

        self.conv2_x = nn.Sequential(
            #TO-DO: Check linear
            nn.Linear(in_features, out_features)
            #nn.Linear(1024*4*4, 3),
            nn.Sigmoid(),
        )

        #TO-DO List: 1\ check conv3_x:upsampling   2\ check parameters of layers        
    def forward(self, x):
        
        x = self.conv1_x(x)
        x = self.conv2_x(x)
        
        return x