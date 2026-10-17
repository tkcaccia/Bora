// Small numerical kernels, not an alternative experimental method.
// Python owns input geometry, statistics, manifests and evaluation.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

static bool solve(std::vector<double> a, std::vector<double> b,
                  std::vector<double>& x, int n) {
    for (int j=0;j<n;++j) {
        int pivot=j;
        for(int i=j+1;i<n;++i)
            if(std::abs(a[i*n+j])>std::abs(a[pivot*n+j])) pivot=i;
        if(std::abs(a[pivot*n+j])<1e-13) return false;
        for(int c=j;c<n;++c) std::swap(a[j*n+c],a[pivot*n+c]);
        std::swap(b[j],b[pivot]);
        double d=a[j*n+j];
        for(int c=j;c<n;++c) a[j*n+c]/=d;
        b[j]/=d;
        for(int i=0;i<n;++i) if(i!=j) {
            double f=a[i*n+j];
            for(int c=j;c<n;++c) a[i*n+c]-=f*a[j*n+c];
            b[i]-=f*b[j];
        }
    }
    x=b; return true;
}

extern "C" int bora_evidence_openmp() {
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}

extern "C" int bora_linear_logits(const float* x,const double* coef,
    const double* intercept,float* out,int64_t n,int d,int k,int threads) {
    if(n<0||d<1||k<1||threads<1) return -1;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(int64_t i=0;i<n;++i) {
        for(int c=0;c<k;++c) {
            double v=intercept[c];
            for(int j=0;j<d;++j) v+=x[i*d+j]*coef[c*d+j];
            out[i*k+c]=static_cast<float>(v);
        }
    }
    return 0;
}

// Minimize .5*||C^T p-x||^2 + .5*ridge*||p-prior||^2 on the simplex.
// Feasible active-set iterations; the equality multiplier is solved with
// each face. Inactive coordinates enter only on a negative reduced gradient.
extern "C" int bora_simplex_qp(const float* rgb,const float* prior,
    const double* colors,float* out,double* residual,int* iterations,
    int64_t n,int k,double ridge,double tolerance,int max_iter,int threads) {
    if(n<0||k<1||ridge<=0||tolerance<=0||max_iter<1||threads<1) return -1;
    std::vector<double> h(k*k);
    for(int a=0;a<k;++a) for(int b=0;b<k;++b) {
        double v=(a==b?ridge:0);
        for(int j=0;j<3;++j) v+=colors[3*a+j]*colors[3*b+j];
        h[a*k+b]=v;
    }
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(int64_t row=0;row<n;++row) {
        std::vector<double> b(k),p(k,0),gradient(k),z(k,0);
        std::vector<int> active;
        int first=0;
        for(int c=0;c<k;++c) {
            b[c]=ridge*prior[row*k+c];
            for(int j=0;j<3;++j) b[c]+=colors[3*c+j]*rgb[row*3+j];
            if(prior[row*k+c]>prior[row*k+first]) first=c;
        }
        active.push_back(first); p[first]=1;
        double violation=INFINITY; int used=0;
        for(;used<max_iter;++used) {
            int s=static_cast<int>(active.size()),m=s+1;
            std::vector<double> a(m*m,0),rhs(m,0),answer;
            for(int i=0;i<s;++i) {
                rhs[i]=b[active[i]];
                for(int j=0;j<s;++j) a[i*m+j]=h[active[i]*k+active[j]];
                a[i*m+s]=1; a[s*m+i]=1;
            }
            rhs[s]=1;
            if(!solve(a,rhs,answer,m)) break;
            std::fill(z.begin(),z.end(),0);
            for(int i=0;i<s;++i) z[active[i]]=answer[i];
            double alpha=1;
            for(int c:active) if(z[c]<0)
                alpha=std::min(alpha,p[c]/std::max(p[c]-z[c],1e-30));
            for(int c=0;c<k;++c) p[c]+=alpha*(z[c]-p[c]);
            if(alpha<1-1e-12) {
                active.erase(std::remove_if(active.begin(),active.end(),
                    [&](int c){if(p[c]<1e-11){p[c]=0;return true;}return false;}),
                    active.end());
                continue;
            }
            int entering=-1; double smallest=0;
            violation=0;
            for(int c=0;c<k;++c) {
                gradient[c]=-b[c]+answer[s];
                for(int j=0;j<k;++j) gradient[c]+=h[c*k+j]*p[j];
                bool present=std::find(active.begin(),active.end(),c)!=active.end();
                if(present) violation=std::max(violation,std::abs(gradient[c]));
                else if(gradient[c]<smallest){smallest=gradient[c];entering=c;}
            }
            violation=std::max(violation,-smallest);
            if(violation<=tolerance){++used;break;}
            if(entering<0) break;
            active.push_back(entering);
        }
        double sum=0;
        for(int c=0;c<k;++c){p[c]=std::max(p[c],0.);sum+=p[c];}
        for(int c=0;c<k;++c) out[row*k+c]=static_cast<float>(p[c]/sum);
        residual[row]=violation; iterations[row]=used;
    }
    return 0;
}

// Axial semi-Markov Viterbi. Exact short-run costs and a linear long-run tail
// allow arbitrarily long regions; there is no hard maximum tissue width.
extern "C" int bora_semimarkov(const float* unary, uint16_t* labels,
    int rows,int columns,int k,int duration_limit,double penalty,int threads) {
    if(rows<1||columns<1||k<1||duration_limit<1||threads<1) return -1;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(int row=0;row<rows;++row) {
        int dmax=std::min(duration_limit,columns);
        std::vector<double> prefix((columns+1)*k,0),dp((columns+1)*k,INFINITY);
        std::vector<int> start((columns+1)*k,0),previous((columns+1)*k,-1);
        std::vector<double> tail(k,INFINITY);
        std::vector<int> tail_start(k,0),tail_previous(k,-1);
        for(int c=0;c<k;++c) dp[c]=0;
        for(int x=1;x<=columns;++x) for(int c=0;c<k;++c)
            prefix[x*k+c]=prefix[(x-1)*k+c]+unary[(static_cast<int64_t>(row)*columns+x-1)*k+c];
        double slope=penalty*.1/(1+dmax);
        auto cost=[&](int length){return penalty*(1+.1*std::log1p(length));};
        auto best_other=[&](int position,int c,int& chosen) {
            if(position==0){chosen=-1;return 0.;}
            double best=INFINITY;chosen=-1;
            for(int j=0;j<k;++j) if(j!=c && dp[position*k+j]<best)
                {best=dp[position*k+j];chosen=j;}
            return best;
        };
        for(int x=1;x<=columns;++x) {
            for(int c=0;c<k;++c) {
                double best=INFINITY;int run_start=0,last=-1;
                for(int length=1;length<=std::min(dmax,x);++length) {
                    int old=-1,s=x-length;
                    double candidate=best_other(s,c,old)+prefix[x*k+c]-prefix[s*k+c]+cost(length);
                    if(candidate<best){best=candidate;run_start=s;last=old;}
                }
                if(x>=dmax) {
                    int s=x-dmax,old=-1;
                    double candidate=best_other(s,c,old)-prefix[s*k+c]-slope*s;
                    if(candidate<tail[c]){tail[c]=candidate;tail_start[c]=s;tail_previous[c]=old;}
                    candidate=prefix[x*k+c]+slope*x+tail[c]+cost(dmax)-slope*dmax;
                    if(candidate<best){best=candidate;run_start=tail_start[c];last=tail_previous[c];}
                }
                dp[x*k+c]=best;start[x*k+c]=run_start;previous[x*k+c]=last;
            }
        }
        int winner=0;
        for(int c=1;c<k;++c) if(dp[columns*k+c]<dp[columns*k+winner]) winner=c;
        int end=columns;
        while(end>0) {
            int begin=start[end*k+winner],old=previous[end*k+winner];
            for(int x=begin;x<end;++x) labels[static_cast<int64_t>(row)*columns+x]=winner;
            end=begin;winner=old;
        }
    }
    return 0;
}

static void adaptive_block(const float* rgb,const double* coef,const double* intercept,
    uint16_t* labels,int width,int k,int y0,int y1,int x0,int x1,
    int64_t& visited,int64_t& certified,int64_t& dense) {
    ++visited;
    double lo[3]={INFINITY,INFINITY,INFINITY},hi[3]={-INFINITY,-INFINITY,-INFINITY};
    for(int y=y0;y<y1;++y) for(int x=x0;x<x1;++x) for(int j=0;j<3;++j) {
        double v=rgb[(static_cast<int64_t>(y)*width+x)*3+j];
        lo[j]=std::min(lo[j],v);hi[j]=std::max(hi[j],v);
    }
    double center[3],radius[3];
    for(int j=0;j<3;++j){center[j]=(lo[j]+hi[j])/2;radius[j]=(hi[j]-lo[j])/2;}
    int winner=0;double best=-INFINITY;
    for(int c=0;c<k;++c) {
        double score=intercept[c];
        for(int j=0;j<3;++j) score+=coef[c*3+j]*center[j];
        if(score>best){best=score;winner=c;}
    }
    bool certain=true;
    for(int c=0;c<k;++c) if(c!=winner) {
        double lower=intercept[winner]-intercept[c];
        for(int j=0;j<3;++j) {
            double delta=coef[winner*3+j]-coef[c*3+j];
            lower+=delta*center[j]-std::abs(delta)*radius[j];
        }
        if(lower<=1e-7) certain=false;
    }
    if(certain) {
        ++certified;
        for(int y=y0;y<y1;++y) for(int x=x0;x<x1;++x)
            labels[static_cast<int64_t>(y)*width+x]=winner;
        return;
    }
    if(y1-y0<=2||x1-x0<=2) {
        for(int y=y0;y<y1;++y) for(int x=x0;x<x1;++x) {
            int label=0;double value=-INFINITY;
            for(int c=0;c<k;++c) {
                double score=intercept[c];
                for(int j=0;j<3;++j) score+=coef[c*3+j]*rgb[(static_cast<int64_t>(y)*width+x)*3+j];
                if(score>value){value=score;label=c;}
            }
            labels[static_cast<int64_t>(y)*width+x]=label;++dense;
        }
        return;
    }
    int ym=(y0+y1)/2,xm=(x0+x1)/2;
    adaptive_block(rgb,coef,intercept,labels,width,k,y0,ym,x0,xm,visited,certified,dense);
    adaptive_block(rgb,coef,intercept,labels,width,k,ym,y1,x0,xm,visited,certified,dense);
    adaptive_block(rgb,coef,intercept,labels,width,k,y0,ym,xm,x1,visited,certified,dense);
    adaptive_block(rgb,coef,intercept,labels,width,k,ym,y1,xm,x1,visited,certified,dense);
}

extern "C" int bora_adaptive_linear(const float* rgb,const double* coef,
    const double* intercept,uint16_t* labels,int64_t* statistics,
    int height,int width,int k,int threads) {
    if(height<1||width<1||k<1||threads<1) return -1;
    int block=64,ny=(height+block-1)/block,nx=(width+block-1)/block;
    int64_t visited=0,certified=0,dense=0;
    #pragma omp parallel for num_threads(threads) reduction(+:visited,certified,dense) schedule(static)
    for(int i=0;i<ny*nx;++i) {
        int y0=(i/nx)*block,x0=(i%nx)*block;
        adaptive_block(rgb,coef,intercept,labels,width,k,y0,std::min(y0+block,height),
                       x0,std::min(x0+block,width),visited,certified,dense);
    }
    statistics[0]=visited;statistics[1]=certified;statistics[2]=dense;
    return 0;
}
