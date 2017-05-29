require 'loadcaffe'
require 'image'
require 'nn'
require 'cunn'
require 'cudnn'

local ContentLoss, parent = torch.class('nn.ContentLoss', 'nn.Module')

function make_vgg()
    local model = nn.Sequential()

    local function ConvBNReLU(nInputPlane, nOutputPlane)
      model:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
      model:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
      model:add(nn.ReLU(true))
      return model
    end

    local MaxPooling = nn.SpatialMaxPooling

    ConvBNReLU(1,64)
    ConvBNReLU(64,64)
    model:add(MaxPooling(2,2,2,2):ceil())

    ConvBNReLU(64,128)
    ConvBNReLU(128,128)
    model:add(MaxPooling(2,2,2,2):ceil())

    ConvBNReLU(128,256)
    ConvBNReLU(256,256)
    ConvBNReLU(256,256)

    local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
--         if cudnn.version >= 4000 then
--            v.bias = nil
--            v.gradBias = nil
--         else
--            v.bias:zero()
--         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')
   BNInit('fbnn.SpatialBatchNormalization')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end

    return model
end

function ContentLoss:__init()
  parent.__init(self)
  self.loss = 0
  self.crit = nn.MSECriterion()

--  self.cnn = loadcaffe.load('content_net/vgg_deploy.prototxt', 'content_net/VGG_ILSVRC_19_layers.caffemodel', 'cudnn'):cuda()
--    local layer_num = #self.cnn.modules
--    while self.cnn:get(layer_num).name ~= 'relu4_2' do
--        self.cnn:remove(layer_num)
--        layer_num = #self.cnn.modules
--    end

    self.cnn = make_vgg():cuda()
    print(self.cnn)
end

function ContentLoss:updateOutput(input, target)
    self.cnn_tar = self.cnn:forward(target):clone()
    self.cnn_inp = self.cnn:forward(input):clone()
    self.loss = self.crit:forward(self.cnn_inp, self.cnn_tar)
  self.output = input
  return self.loss
end

function ContentLoss:updateGradInput(input, gradOutput)
    local grad = self.crit:backward(self.cnn_inp, self.cnn_tar)
--    self.gradInput = self.cnn:backward(input, grad)
    self.gradInput = self.cnn:updateGradInput(input, grad)

  self.gradInput:add(gradOutput)
    collectgarbage()
  return self.gradInput
end


