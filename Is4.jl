mutable struct Site
    snum
    spin
    vec
    z
end
Ldim=10;
J=-1;
# snap=200;
sites=Array{Site}(undef,Ldim,Ldim)

function initsys()
    #sites=Matrix{Site}(undef,Ldim,Ldim)
    for i=1:Ldim*Ldim
        if mod(i, Ldim)==1
            v1=i+Ldim-1     #v1=-1 for Non-PBC
        else
            v1=i-1
        end
        if i>=Ldim*(Ldim-1)+1
            v2= mod(i, Ldim)==0 ? Ldim : mod(i, Ldim);
        else
            v2=i+Ldim
        end
        if mod(i, Ldim)==0
            v3=i-Ldim+1     #v1=-1 for Non-PBC
        else
            v3=i+1
        end
        if i<=Ldim
            v4= Ldim*(Ldim-1)+i;
        else
            v4=i-Ldim
        end

        global sites[i]=Site(i,rand([-1,1]),[v1 v2 v3 v4],0)
    end
end

function hamil()
    en=0;
    for i=1:Ldim*Ldim, j=1:4
        en=en+J*sites[i].spin*sites[sites[i].vec[j]].spin
    end
    return en/2
end

function dH(i, spinOLD, spinNEW)
    en=0;
    for j=1:4
        en=en+J*(spinNEW-spinOLD)*sites[sites[i].vec[j]].spin;
    end
    return en;
end

function MC(T)
    s=rand(collect(1:Ldim*Ldim));
    sspin=sites[s].spin;
    if sspin>0
        sites[s].spin=-1
    else
        sites[s].spin=1
    end
    cluster=[s];
    stack=[s];
    for i=1:4
        if sites[sites[s].vec[i]].spin==sspin
            r=rand()
            if r<1-exp(2*J/T)
                append!(cluster, sites[s].vec[i])
                append!(stack, sites[s].vec[i])
                if sspin>0
                    sites[sites[s].vec[i]].spin=-1
                else
                    sites[sites[s].vec[i]].spin=1
                end
            end
        end
    end
    popfirst!(stack)
    while length(stack)>0
        s2=stack[1]
        newvec=setdiff(sites[s2].vec, cluster)
        for i=1:length(newvec)
            if sites[newvec[i]].spin==sspin
                r=rand()
                if r<1-exp(2*J/T)
                    append!(cluster, newvec[i])
                    append!(stack, newvec[i])
                    if sspin>0
                        sites[newvec[i]].spin=-1
                    else
                        sites[newvec[i]].spin=1
                    end
                end
            end
        end
        #println("stack ", stack)
        #println("cluster ", cluster)
        popfirst!(stack)
    end
end

meanspin=[0.0];
spinM=[0.0];
spinMM=[0.0];

function mSpin()
    mS=0.0
    for i=1:Ldim*Ldim
        mS=mS+sites[i].spin;
    end
    mS=mS/(Ldim*Ldim);

    append!(meanspin, mS);

end

function mnsp(snap)

    mn=sum(meanspin[i] for i=2:snap+1)/snap
    append!(spinM, mn)
    mnn=sqrt(sum((meanspin[i]-mn)^2 for i=2:snap+1)/snap)
    append!(spinMM, mnn)
    global meanspin=[0.0]
end

function mSpin2(xArr)

    for i=1:Ldim*Ldim
        append!(xArr,sites[i].spin)
    end
end

function prog2(xArr, MCS, snap)
    T=1.6;
    counter=0
    MCsteps=MCS;
    for i=1:MCsteps
        MC(T)
        if i>MCsteps*0.9 && counter<snap
        # if i>MCsteps*0.9 && counter<snap && mod(i,10^0)==0
            mSpin2(xArr)
            mSpin()
            counter=counter+1
        end
    end
    mnsp(snap);
    while T<2.9
        T=T+0.1
        counter=0
        println(T)
        for j=1:MCsteps
            MC(T)
            if j>MCsteps*0.9 && counter<snap
            # if j>MCsteps*0.9 && counter<snap && mod(j,10^0)==0
                mSpin2(xArr)
                mSpin()
                counter=counter+1
            end
        end
        mnsp(snap)
    end
end

function main2(xArr, MCS, snap)
    initsys()
    global meanspin=[0.0];
    global spinM=[0.0];
    global spinMM=[0.0];
    prog2(xArr, MCS, snap);
    popfirst!(spinM)
    popfirst!(spinMM)

end

function callFunc(MCS, snap, L)
    global Ldim = L
    global sites=Array{Site}(undef,Ldim,Ldim)
    xArr=[0.0];
    main2(xArr, MCS, snap)
    popfirst!(xArr)
    TRange=length(spinM)
    xRes=reshape(xArr,Int(Ldim*Ldim),Int(length(xArr)/(Ldim*Ldim)))'
    mean2=[xRes[j,i]-sum(xRes[:,i])/(TRange*snap) for j=1:(TRange*snap),i=1:Ldim*Ldim]
    return mean2
end

# r = callFunc(1000,100,20)
#
# using Interact, Images
#
# @manipulate for i in 1:1400
#     colorview(Gray, reshape(r[i,:],Int(sqrt(size(r,2))),Int(sqrt(size(r,2)))))
#     # colorview(Gray, reshape(r[i,:],20,20))
# end
