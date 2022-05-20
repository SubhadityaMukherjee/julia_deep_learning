import CairoMakie
using FastAI
using Flux

dir = "/media/hdd/Datasets/asl/asl_alphabet_train/"
images = Datasets.loadfolderdata(
    dir,
    filterfn=isimagefile,
    loadfn=loadfile)

getlabel(x) = pathname(pathparent(x))
labels = Datasets.loadfolderdata(
    dir,
    filterfn=isimagefile,
    loadfn=getlabel)

data = (images, labels)
nobs(data)

image, lb = sample = getobs(data, 1);
image
lb

classes = readdir(dir);

traindata, valdata = splitobs(shuffleobs(data), at = 0.8);
blocks = (Image{2}(), Label(classes))
task = ImageClassificationSingle(blocks, (128, 128));

dls = taskdataloaders(traindata, task, 128);
model = taskmodel(task, Models.xresnet18());
lossfn = Flux.Losses.logitcrossentropy;
opt = ADAM()
# learner = Learner(model, dls, opt, lossfn, ToGPU(), Metrics(accuracy));
learner = tasklearner(task, traindata, callbacks=[ToGPU(), Metrics(accuracy)])
fitonecycle!(learner, 4)
fitonecycle!(learner, 10)
# showoutputs(task, learner)