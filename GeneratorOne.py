class GeneratorOne(nn.Module):
    def __init__(self, block):
        super(GeneratorOne, self).__init__()
        
        self.conv1_x = nn.Sequential(                   #Down-sampling component
            nn.Conv2d(3, 16, 3, padding = 1),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(3, 16, 3, padding = 1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(16),

            nn.Conv2d(3, 16, 3, padding = 1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(16),

            nn.Conv2d(3, 16, 3, padding = 1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(16),

            nn.Conv2d(3, 16, 3, padding = 1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(16),

            nn.Conv2d(3, 16, 3, padding = 1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(16),

            nn.Conv2d(3, 16, 3, padding = 1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(16)
        )
            
        self.conv2_x = nn.Sequential(                   #Up-sampling component
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True),
            nn.Conv2d(16, 32, (4, 4), stride = 2, padding = "same"),
            nn.BatchNorm2d(32),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True),
            nn.Conv2d(16, 32, (4, 4), stride = 2, padding = "same"),
            nn.BatchNorm2d(32),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True),
            nn.Conv2d(16, 32, (4, 4), stride = 2, padding = "same"),
            nn.BatchNorm2d(32),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True),
            nn.Conv2d(16, 32, (4, 4), stride = 2, padding = "same"),
            nn.BatchNorm2d(32),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True),
            nn.Conv2d(16, 32, (4, 4), stride = 2, padding = "same"),
            nn.BatchNorm2d(32),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True),
            nn.Conv2d(16, 32, (4, 4), stride = 2, padding = "same"),
            nn.BatchNorm2d(32),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True),
            nn.Conv2d(16, 32, (4, 4), stride = 2, padding = "same"),
            nn.BatchNorm2d(32)
        )

        self.conv3_x = nn.Sequential(                   #Tanh active component
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True),
            nn.Tanh()
        )    
        #TO-DO List: 1\ check conv3_x:upsampling   2\ check parameters of layers        
    def forward(self, x):
        
        x = self.conv1_x(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        
        return x