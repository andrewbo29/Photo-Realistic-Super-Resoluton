require 'loadcaffe'
require 'image'

local ContentLoss, parent = torch.class('nn.ContentLoss', 'nn.Module')

function ContentLoss:__init()
  parent.__init(self)
--  self.target = target or torch.Tensor()
  self.loss = 0
  self.crit = nn.MSECriterion()
  self.active = true

  self.cnn = loadcaffe.load('content_net/vgg_deploy.prototxt', 'content_net/VGG_ILSVRC_19_layers.caffemodel', 'cudnn'):cuda()
    local layer_num = #self.cnn.modules
    while self.cnn:get(layer_num).name ~= 'relu4_2' do
        self.cnn:remove(layer_num)
        layer_num = #self.cnn.modules
    end
end

function ContentLoss:updateOutput(input, target)
--  if input:nElement() == self.target:nElement() and self.active then
--    self.loss = self.crit:forward(input, self.target) * self.strength
--  end

    input = image.scale(input, 224, 224)
    target = image.scale(target, 224, 224)

    self.cnn_tar = self.cnn:forward(target)
    self.cnn_inp = self.cnn:forward(input)
    self.loss = self.crit:forward(self.cnn_inp, self.cnn_tar)
  self.output = input
  return self.output
end

function ContentLoss:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(gradOutput)

  if input:nElement() == self.target:nElement() and self.active then
    self.gradInput = self.crit:backward(self.cnn_inp, self.cnn_tar)
  else
    self.gradInput:zero()
  end

  self.gradInput:add(gradOutput)
  return self.gradInput
end


