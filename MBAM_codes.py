import autograd as au
import autograd.numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.integrate import odeint
def N_sphere(F):
    N=F.shape[0]+1
    A=np.zeros((N,)+F.shape)+1
    for i in range(N-1):
        A[i,i]=np.cos(F[i])
        A[(i+1):N,i,:]=np.sin(F[i])
    
    return np.prod(A,axis=1)[::-1,:]
class diff_FIM:
    def __init__(self,t:"x-axis measurements", y:"measurements",σ:"errors",model:"model function"):
        self.t  = t
        self.y  = y
        self.f  = model
        self.σ  = σ
        self.J  = au.jacobian(lambda θ:(y-model(t,θ))/σ)#,argnum=1)
        self.H  = au.hessian(lambda θ:(y-model(t,θ))/σ)#,argnum=1)
    def _r(self, θ:"Model parameters")->"Residuals":
        return (self.y-self.f(self.t,θ))/self.σ
    def χ2(self, θ:"Model parameters")->"χ^2 value":
        return np.sum((self.y-self.f(self.t,θ))**2/self.σ**2)
    def g(self, θ:"Model parameters")->"FIM for parameters θ":
        J=self.J(θ)
        return np.einsum('im,in',J,J)
    def eigval(self, θ:"Model parameters",k=0)->"kth eigenvalue for parameters θ":
        g = self.g(θ)
        return np.linalg.eigh(g)[0][k]
    def eigvector(self, θ:"Model parameters",k=0)->"kth eigenvalue for parameters θ":
        g = self.g(θ)
        return np.linalg.eigh(g)[1][:,k]
    def detg(self,θ:"Model parameters")->"determinant of FIM":
        return np.linalg.det(self.g(θ))
    def signature(self,θ:"Model parameters")->"Metric signature":
        return np.sum(np.sign(np.linalg.eigvals(self.g(θ))))
    def ginv(self,θ:"Model parameters")->"Metric inverse":
        J=self.J(θ)
        return np.linalg.inv(np.einsum('im,in',J,J))
    def Γ1(self,θ:"Model parameters")->"Christoffel symbols of the first kind":
        J=self.J(θ)
        H=self.H(θ)
        return np.einsum('im,iab',J,H)
    def Γ2(self,θ:"Model parameters")->"Christoffel symbols of the second kind":
        return np.einsum('ij,jab',self.ginv(θ),self.Γ1(θ))
    def P(self,θ:"Model parameters")->"Residual space normal projection operator":
        J=self.J(θ)
        g=self.g(θ)
        P1=np.einsum('mn,im,jn',g,J,J)
        return 1-P1
    def Riemann(self,θ:"Model parameters")->"Riemann tensor":
        H=self.H(θ)
        P=self.P(θ)
        return np.einsum('ima,ij,jbn->mnab',H,P,H)-np.einsum('imb,ij,jan->mnab',H,P,H)

    def Ricci(self,θ:"Model parameters")->"Ricci tensor":
        H=self.H(θ)
        g=self.g(θ)
        P=self.P(θ)
        return np.einsum('iab,ab,ij,jmn->mn',H,g,P,H)-np.einsum('ins,ij,as,jam->mn',H,P,g,H)
    def Ricci_R(self,θ:"Model parameters")->"Ricci curvature scalar":
        H=self.H(θ)
        g=self.g(θ)
        P=self.P(θ)
        return np.einsum('iam,ma,ij,jbn,bn',H,g,P,H,g)-np.einsum('ims,ij,jab,ma,sb',H,P,H,g,g)
    def external_v2(self,θ:"Model parameters",dθ:"derivatives of model parameters")->"Residual space velocity^2":
        g=self.g(θ)
        return np.einsum('m,mn,n',dθ,g,dθ)
    def external_a2(self, θ:"Model parameters",dθ:"derivatives of model parameters")->"Residual space acceleration^2":
        H=self.H(θ)
        P=self.P(θ)
        a=np.einsum('imn,ij,jab',H,P,H)
        return np.einsum('m,n,a,b,mnab',dθ,dθ,dθ,dθ,a)
    def external_R(self, θ:"Model parameters",dθ:"derivatives of model parameters")->"Extrinsic curvature radius":
        v2=self.external_v2(θ,dθ)
        a2=self.external_a2(θ,dθ)
        return v2/np.sqrt(a2)
    def external_ω(self,θ:"Model parameters",dθ:"derivatives of model parameters")->"Extrinsic frequency":
        v2=self.external_v2(θ,dθ)
        a2=self.external_a2(θ,dθ)
        return np.sqrt(np.abs(v2*a2))
    def external_ωv(self,θ:"Model parameters",dθ:"derivatives of model parameters")->"Extrinsic normalized frequency":
        v2=self.external_v2(θ,dθ)
        a2=self.external_a2(θ,dθ)
        return np.sqrt(np.abs(a2))
    def calc_ω0(self,θ:"Model parameters",k:"index of the k-th smallest eigenvalue of g")->"Extrinsic frequency in the k-th eigendirection":
        g=self.g(θ)
        w,v=np.linalg.eigh(g)
        vn=v[:,k]
        return self.external_ω(θ,vn)/(2*np.pi)
    def find_tetrad(self,θ:"Model parameters")->"Finds tetrad and inverse tetrad of FIM":
        v,e0=np.linalg.eig(self.g(θ))
        e=np.einsum('ia,ab->ib',e0,np.diag(np.sqrt(v)))
        einv=np.einsum('ia,ab->ib',e0,np.diag(1/np.sqrt(v)))
        return e,einv
    def compute_sphere(self,θ:"Model parameters",Npoints:"Mesh size"=10)->"Computes points on a sphere of unit radius":
        g       = self.g(θ)
        e,einv  = self.find_tetrad(θ)
        ndim    = θ.shape[0]
        angles  = [np.linspace(0,np.pi,Npoints) for i in range(ndim-2)]
        angles.append(np.linspace(0,2*np.pi,Npoints))
        Angles  = np.meshgrid(*angles)
        coord   = N_sphere(np.array([A.flatten() for A in Angles]))
        return np.einsum('ia,ab->ib',einv,coord),coord,Angles
    def compute_ω_sphere(self,θ:"Model parameters",Npoints:"Mesh size"=10)->"Computes normalized frequency for each point on a sphere":
        c, c0,A = self.compute_sphere(θ,Npoints)
        ωv      = np.array([self.external_ωv(θ,c[:,i]) for i in range(c.shape[1])])
        return c, c0,ωv,A
    
    def delete_offending_index(self,tensor:"tensor of any rank",i:"which dimension to remove")->"minor of the tensor":
        Naxis = np.size(tensor.shape)
        NewT  = tensor
        for j in range(Naxis):
            NewT = np.delete(NewT,i,axis=j)
        return NewT
    def find_offending_index(self,g:"Metric tensor")->"Finds minor of FIM without which the FIM has a full rank":
        N      = g.shape[0]
        cut    = np.array([np.linalg.matrix_rank(self.delete_offending_index(g,i)) 
                               for i in range(N)])==(N-1)
        if np.sum(cut)==0:
            raise Warning('Nothing found by removing 1 axis')
        return np.argmax(cut)
    def find_MBAM_IC(self,θ:"Model parameters",k:"Index of the eigendirection"=0)->"Produces initial conditions for the geodesic equation":
        g = self.g(θ)
        N = g.shape[0]
        if np.linalg.matrix_rank(g)==N:
            λ,v    = np.linalg.eigh(g)
            V = v[:,k]
            θ2   = np.einsum('i,j,ij',θ, θ,g)
            v2   = np.einsum('i,j,ij',V,V,g)  
        elif k<N-1:
            i = self.find_offending_index(g)
            gnew = self.delete_offending_index(g,i)
            θnew = self.delete_offending_index(θ,i)
            λ,v    = np.linalg.eigh(gnew)
            θ2   = np.einsum('i,j,ij',θnew, θnew,gnew)
            v2   = np.einsum('i,j,ij',v[:,k],v[:,k],gnew)
            v = v[:,k]
            V = np.append(v[:i],[0])
            V = np.append(V,v[i:])
        else:
            i      = self.find_offending_index(g)
            V      = np.zeros(N)
            V[i]   = 1
            gnew = self.delete_offending_index(g,i)
            θnew = self.delete_offending_index(θ,i)
            λ,v    = np.linalg.eigh(gnew)
            θ2   = np.einsum('i,j,ij',θnew, θnew,gnew)
            v2   = np.einsum('i,j,ij',v[:,0],v[:,0],gnew)
        τ = np.sqrt(θ2/v2)
        return np.append(θ, V),τ

    def MBAM_RHS(self,V:"2N dimensional initial conditions vector")->"RHS of the geodesic equation":
        N   = int(np.size(V)/2)
        θ   = V[:N]
        dθ  = V[N:]
        g   = self.g(θ)
        ret = np.array([dθ])
        if np.linalg.matrix_rank(g)==N:
            
            ret = np.append(ret,-np.einsum('a,b,cab->c',dθ,dθ,self.Γ2(θ)))
        else:
            return np.array(2*N*[np.nan])
            #i      = self.find_offending_index(g)
            #gnew   = self.delete_offending_index(g,i)
            #Γ1new  = self.delete_offending_index(self.Γ1(θ),i)
            #Γ2new  = np.einsum('ij,jab->iab',np.linalg.inv(gnew),Γ1new)
            #dθnew  = self.delete_offending_index(dθ,i)
            #ddθnew = -np.einsum('a,b,cab->c',dθnew,dθnew,Γ2new)
            #ret    = np.append(ret,ddθnew[:i])
            #ret    = np.append(ret,[0])
            #ret    = np.append(ret,ddθnew[i:])
        
        return ret
    def run_MBAM(self,θ:"Model parameters",k:"Initial eigendirection"=0,dmax=10,T=0)->"computes the geodesic equation":
        def fun(V,t):
            return self.MBAM_RHS(V)
        V0,τ = self.find_MBAM_IC(θ,k)
        if T is None:
            T    = np.linspace(0,10*τ,100)#np.logspace(np.log10(τ),dmax+np.log10(τ),100)
        return τ, T,odeint(fun,V0,T)
    
    
def SymLogNorm(dg:"Values to plot w/ pcolormesh")->"Matplotlib lognorm":
    adg  = np.abs(dg)
    vmax = adg.max()
    vmin = adg.min()
    vmin = np.max([10**(-5.)*vmax, vmin])
    if np.sum(dg<0)>0:
        return colors.SymLogNorm(linthresh=vmin,vmin=-vmax, vmax=vmax)
    else:
        return colors.LogNorm(vmin=vmin, vmax=vmax)

class MBAM_plotting(diff_FIM):
    def __init__(self,model:"Model to analyze",NAME:"string", X:"X data", Y:"Y data", σY:"Error bars",θ_bf:"Best fitting parameters",xlim:"x values limit",ylim:"y values limit"):
        self.NAME = NAME
        self.xlim = xlim
        self.ylim = ylim
        self.N_D  = len(X)
        self.N_P  = len(θ_bf)
        diff_FIM.__init__(self,X,Y,σY,model)
        self.θ_bf = θ_bf
    
    def plot_data(self,ax:"matplotlib axis",color:"color of data points"='C0',fit:"show a fit"=True)->"points and residuals plotted":
        ax[0].set_xlim(self.t.min(),self.t.max())
        ax[1].set_xlim(self.t.min(),self.t.max())
        ax[0].set_ylim(self.y.min(),self.y.max())
        ax[1].set_ylim(-2,2)
        ax[0].errorbar(self.t,self.y,yerr=self.σ,fmt=color+'o',capsize=2)
        if fit:
            ax[0].plot(self.t,self.f(self.t,self.θ_bf),'k',lw=2)
        ax[0].set_xlabel('$t$')
        ax[0].set_ylabel('$y(t)$')
        ax[1].errorbar(self.t,self._r(self.θ_bf),yerr=self.N_D*(1,),fmt='o',capsize=2)
        ax[1].axhline(0,color='k')
        ax[1].set_xlabel('$t$')
        ax[1].set_ylabel('$r(t)$')
  

    def construct_sphere(self,θ:"Model parameters"=None)->"Plots an unit sphere":
        if θ is None:
            θ = self.θ_bf
        c,c0,ωv,A=self.compute_ω_sphere(θ,Npoints=10)
        w,v=np.linalg.eigh(self.g(θ))
        f,ax=plt.subplots(1,3,figsize=(23,7))
        ax[0].plot(c[0],c[1],'k',ls='-',zorder=0)
        ax[1].plot(c0[0],c0[1],'k',zorder=0)
        im0=ax[0].scatter(c[0],c[1],c=ωv/(2*np.pi),zorder=1,s=10**2)
        ax[0].set_title('Standardni sustav')
        ax[1].set_title('Lokalni inercijalni sustav')
        im1=ax[1].scatter(c0[0],c0[1],c=ωv/(2*np.pi),zorder=1,s=10**2)
        e,ei=self.find_tetrad(self.θ_bf)
        for i in range(2):
            vn=v[:,i]
            vn=vn/np.sqrt(np.einsum('i,ij,j',vn,self.g(θ),vn))
            vninv=np.einsum('im,m',e,vn)
            ax[0].plot([0,vn[0]],[0,vn[1]],
                       color='C%d'%i,lw=3,
                       label='$\lambda=%e$\n'%w[i]+'$\omega=%e$'%self.external_ωv(θ,vn))
            ax[1].plot([0,vninv[0]],[0,vninv[1]],'C%d'%i,lw=3,)
        ax[0].legend(loc=1)
        f.colorbar(im0,ax=ax[0]).set_label('$\omega/v/(2\pi)$')
        f.colorbar(im1,ax=ax[1]).set_label('$\omega/v/(2\pi)$')
        ax[0].set_xlabel(r'$\dot\theta^{\mu=0}$')
        ax[0].set_xlabel(r'$\dot\theta^{\mu=1}$')
        ax[1].set_xlabel(r'$\dot\theta^{i=0}$')
        ax[1].set_xlabel(r'$\dot\theta^{i=1}$')
        ax[2].plot(A[0],ωv/(2*np.pi))
        ax[2].set_xlabel(r'$\varphi^0$')
        ax[2].set_ylabel('$\omega/v/(2\pi)$')
        f.tight_layout()
        f.savefig(NAME+'_omegas.pdf')
        return f
    def plot_scalar(self,ax:"Matplotlib axis",X:"X mesh to plot",Y:"Y mesh to plot",Z:"Colors to plot",Vx:"X compontents of vectors to plot"=None,Vy:"Y compontents of vectors to plot"=None,xlabel=r'$\theta^0$',ylabel=r'$\theta^1$',veccolor='k'):
        ax.plot(*self.θ_bf,'ro')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        im=ax.pcolormesh(X,Y,Z,norm=SymLogNorm(Z.flatten()),
                            cmap='gray')
        plt.colorbar(im,ax=ax)
        ax.set_xlim(*self.xlim)
        ax.set_ylim(*self.ylim)
        if Vx is not None:
            ax.streamplot(X,Y,Vx,Vy,color=veccolor)
    def construct_mesh(self,N:"Produces a mesh of dimension N")->"List of x and y points and the corresponding meshgrid":
        θ1 = np.linspace(*self.xlim,N)
        θ2 = np.linspace(*self.ylim,N)
        X,Y  = np.meshgrid(θ1,θ2)
        return θ1, θ2, X, Y
    def apply_on_mesh(self,N:"Mesh dimension",F:"List of functions to plot")->"parameters, mesh and evaluations of functions on the mesh":
        θ1, θ2, X, Y = self.construct_mesh(N)
        return θ1, θ2, X, Y,np.array([[ [f(np.array([t1,t2])) for f in F] for t1 in θ1] for t2 in θ2])

    def bar_plot(self,ax:"Axis to plot on",*args:"Vectors to plot",xlabel='',ylabel='',plabels:"x tick labels"=None,labels:"Labels of vectors"=None,colors:"Vector colors"=None,lw=1)->"Adds a plot of eigenvectors as bar plots, filled if a component is positive, white if negative":
        Nv = len(args)
        bw = .5/Nv
        for i in range(Nv):
            v  = args[i]
            if labels is not None and i<len(labels):
                label = labels[i]
            else:
                label = ""
            if colors is not None and i<len(colors):
                color = colors[i]
            else:
                color = "k"
            Nθ = np.size(v)
            ax.bar(np.arange(Nθ)+bw,np.abs(v),.8/Nv, 
                   color=np.where(v>0,color,"1.0"),
                   edgecolor=color,label=label,lw=1)
            bw+=1/Nv
        ax.set_xticks(np.arange(0.5,Nθ+.5,1))
        ax.set_yscale('log')
        if plabels is not None:
            ax.set_xticklabels(plabels)
        else:
            ax.set_xticklabels([r"$\theta^{"+"%d"%i+"}$" for i in range(Nθ)])
        if labels is not None:
            ax.legend()
