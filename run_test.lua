require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'nngraph'
require 'cudnn'
util = paths.dofile('util.lua')
torch.setdefaulttensortype('torch.FloatTensor')

opt = {
  dataset = 'folder', 
  batchSize=32,
  niter=1,
  ntrain = math.huge, 
  gpu=1,
  nThreads = 4,
  scale=4,
  loadSize=96,
  t_folder='',
  model_file='models/modelssuper_resolution_adversarial_G_9',
  result_path='',
  test_path='',
  test='true'
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

local DataLoader = paths.dofile('data/data.lua')
data = DataLoader.new(opt.nThreads,  opt)

modelG=util.load(opt.model_file,opt.gpu)

print('Generating image')

cnt=1
for i = 1, opt.niter do
  _,input= data:getBatch()
  fake = modelG:forward(input)
  fake[fake:gt(1)]=1
  fake[fake:lt(0)]=0

  for j=1,opt.batchSize do
    local fname = string.format('%s/fake_%04d.png',opt.result_path,cnt)
    image.save(fname,image.toDisplayTensor(fake[j]))
    print('Generated in ' .. fname)
    cnt=cnt+1
  end
end




