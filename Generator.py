class Generator(nn.Module):
    def __init__(self, block):
        super(Generator, self).__init__()
        
        self.conv1_x = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2))
        
        self.conv2_x = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2))

        self.conv3_x = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))

        self.conv1_y = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2))
        
        self.conv2_y = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2))

        self.conv3_y = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))      
        
#         self.conv1_x = nn.Conv2d(3, 16, 3, padding = 1)
#         self.conv2_x = nn.Conv2d(16, 32, 3)
#         self.conv3_x = nn.Conv2d(32, 16, 3, padding=1)

#         self.conv1_y = nn.Conv2d(3, 16, 3)
#         self.conv2_y = nn.Conv2d(16, 32, 3)
#         self.conv3_y = nn.Conv2d(32, 16, 3, padding = 1)

#         self.relu = nn.ReLU()
#         self.avgpool = nn.AvgPool2d(2)
        
    
        # 2 Residual Blocks for Identity Image
        self.block1_x = block(16, 16)
#         downsample_x = nn.Sequential(conv3x3(16, 1, 1), nn.BatchNorm2d(1))
#         self.block2_x = block(16, 1, 1, downsample_x)
        self.block2_x = block(16, 16)

        # 2 Residual Blocks for Shape Image
        self.block1_y = block(16, 16)
#         downsample_y = nn.Sequential(conv3x3(16, 1, 1), nn.BatchNorm2d(1))
#         self.block2_y = block(16, 1, 1, downsample_y)
        self.block2_y = block(16, 16)
        # 2 Residual Blocks for Combined(concat) image
        downsample1_concat = nn.Sequential(conv3x3(32, 16, 1), nn.BatchNorm2d(16))
        self.block1_concat = block(32, 16, 1, downsample1_concat)

        self.block2_concat = block(16, 16)
        
#         self.deconv1 = nn.ConvTranspose2d(16, 16, 3)
#         self.deconv2 = nn.ConvTranspose2d(16, 3, 3)
        
    
#          self.conv2_y = nn.Sequential(
#             nn.Conv2d(16, 32, 3, padding = 1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.AvgPool2d(2))
    
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True),
            nn.ConvTranspose2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(32, 3, 3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True))
        
        
    def forward(self, x, y):
        
        x = self.conv1_x(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.block1_x(x)
        x = self.block2_x(x)
        
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