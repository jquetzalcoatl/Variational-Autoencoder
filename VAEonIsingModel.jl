using Flux, Plots, Images, Interact
include("Is4.jl")

#callfunc takes # of MCS, # of snapshots, system linear size, ~ (10^4,100,20)
MCS = 10000
snap = 300
lsize = 30
X = callFunc(MCS, snap, lsize)
@manipulate for nbinsize in 0.1:0.1:2
  histogram(reshape(X, length(X),1), nbins=minimum(X):nbinsize:maximum(X))
end
Y = [(X[i,j] - minimum(X[i,:]))/(maximum(X[i,:]) - minimum(X[i,:])) for i in 1:size(X,1), j in 1:size(X,2) ]
@manipulate for nbinsize in 0.1:0.1:2
  histogram(reshape(Y, length(Y),1), nbins=minimum(Y):nbinsize:maximum(Y))
end

X = Y'

N, M = size(X, 2), 20
data = [X[:,i] for i in Iterators.partition(1:N,M)]

f1 = Dense(size(X, 1),400, celu)

f21, f22 = Dense(400,2), Dense(400,2)

f3 = Chain(Dense(2,400, celu), Dense(400,size(X, 1), σ))

function encode(x)
  h = f1(x)
  μ, logσ = f21(h), f22(h)
  z = μ + exp.( logσ) .* randn(size(logσ,1))
  h3 = f3(z)
  return h3, μ, logσ
end

kl_q_p(μ, logσ) = 0.5f0 * sum(exp.(2f0 .* logσ) + μ.^2 .- 1f0 .- (2 .* logσ))

loss(x) = ((h3, μ, logσ) = encode(x); Flux.mse(h3,x) + (kl_q_p(μ, logσ)) * 1 // M )
# loss(x) = ((h3, μ, logσ) = encode(x); Flux.mse(h3,x) + 0.001f0 * sum(x->sum(x.^2), params(f1,f21,f22,f3))  + 1.0*(kl_q_p(μ, logσ)) * 1 // M )
# loss(x) = ((h3, μ, logσ) = encode(x); Flux.mse(h3,x)  )
plot(loss.(data))

opt = ADAM(0.0001, (0.9, 0.8))

ps = params(f1,f21,f22,f3)

# @progress for i = 1:20
#   @info "Epoch $i"
#   Flux.train!(loss, ps, data[1:20], opt)
# end

# my_custom_train!(loss, ps, data[1:20], opt)

@progress for i = 1:20
  @info "Epoch $i"
  # Flux.train!(loss, ps, zip(data), opt, cb=evalcb)
  plt=my_custom_train!(loss, ps, data, opt)
  display(plt)
end

function my_custom_train!(loss, ps, data, opt)
  ps = Flux.Params(ps)
  lost_list = []
  plt=scatter(legend=false)
  plt = plot!(loss.(data))
  for x in data
    gs = gradient(() -> loss(x), ps)
    # gs = gradient(ps) do
    #   training_loss = loss(d...)
    #   # Insert what ever code you want here that needs Training loss, e.g. logging
    #   return training_loss
    # end
    # insert what ever code you want here that needs gradient
    # E.g. logging with TensorBoardLogger.jl as histogram so you can see if it is becoming huge
    Flux.update!(opt, ps, gs)
    # Here you might like to check validation set accuracy, and break out to do early stopping
    append!(lost_list, loss(x))
    # println(lost_list)
    plt=scatter!(lost_list)
  end
  plt=hline!([0])
  return plt
end

loss.(data)
data

h = f1(X)
μ, logσ = f21(h), f22(h)
z = μ + exp.( logσ) .* randn(size(logσ,1))
h3 = f3(z)


@manipulate for n in 1:size(z,2)
  T = 1.6 + 0.1*floor(n/snap)
  plot(legend=false, xlim=(minimum(z[1,:]), maximum(z[1,:])), ylim=(minimum(z[2,:]), maximum(z[2,:])))
  scatter!(z[1,1:n],z[2,1:n], mz=1:size(z,2), mc=:rainbow, ms=5,ma=0.6, title="$T", xlabel="z1", ylabel="z2")
end

@manipulate for i=1:size(h3,2)
    colorview(Gray,reshape(h3[:,i],lsize,lsize))
end

@manipulate for i=1:size(X,2)
    colorview(Gray,reshape(X[:,i],lsize,lsize))
end

n=3000
T = 1.6 + 0.1*floor(n/snap)
plt = scatter(z[1,1:n],z[2,1:n], mz=1:size(z,2), mc=:rainbow, ms=5,ma=0.6, title="Temperature = $T", xlabel="z1", ylabel="z2", legend=:false)
savefig(plt, pwd()*"/Flux/LatentSpaceT>Tc.png")

i=3000
im = colorview(Gray,reshape(X[:,i],lsize,lsize))

save(pwd()*"/Flux/LatentSpaceImT>Tc.png", im)
