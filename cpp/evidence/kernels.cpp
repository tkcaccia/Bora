// Small numerical kernels, not an alternative experimental method.
// Python owns input geometry, statistics, manifests and evaluation.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <queue>
#include <unordered_set>
#include <tuple>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

// Gray-level geodesic reconstruction: exact widest-path closure under a ceiling.
// Classes are independent; use at most the caller's two OpenMP threads.
extern "C" int bora_gray_reconstruction(const double* ceiling,const double* marker,
        double* output,int h,int w,int k,int threads) {
    if(h<1||w<1||k<1||threads<1)return 1;
    const int n=h*w;
    for(int i=0;i<n*k;++i)if(!std::isfinite(ceiling[i])||!std::isfinite(marker[i])||
        ceiling[i]<0||marker[i]<0||marker[i]>ceiling[i]+1e-12)return 2;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(int g=0;g<k;++g) {
        std::priority_queue<std::pair<double,int>> queue;
        for(int v=0;v<n;++v){output[v*k+g]=marker[v*k+g];if(marker[v*k+g]>0)queue.emplace(marker[v*k+g],v);}
        while(!queue.empty()) {
            auto item=queue.top();queue.pop();double value=item.first;int v=item.second;
            if(value<output[v*k+g])continue;
            int x=v%w,y=v/w;
            const int adjacent[4]={x>0?v-1:-1,x+1<w?v+1:-1,y>0?v-w:-1,y+1<h?v+w:-1};
            for(int u:adjacent)if(u>=0) {
                double next=std::min(value,ceiling[u*k+g]);
                if(next>output[u*k+g]){output[u*k+g]=next;queue.emplace(next,u);}
            }
        }
    }
    return 0;
}

// Greedy positive shared-class Bayes factors. This certifies the specified
// greedy operation, not global partition optimality or independent cells.
extern "C" int bora_bayes_regions(const double* log_probability,const uint8_t* support,
        int32_t* output,int h,int w,int k,double* diagnostics) {
    if(h<1||w<1||k<2)return 1;
    int n=h*w;std::vector<int> parent(n),size(n,1),version(n,0),compact(n,-1);
    std::vector<double> sums(log_probability,log_probability+n*k),evidence(n,0.);
    std::vector<std::unordered_set<int>> neighbours(n);
    for(int i=0;i<n*k;++i)if(!std::isfinite(sums[i]))return 2;
    for(int v=0;v<n;++v)parent[v]=v;
    auto root=[&](int v){while(parent[v]!=v){parent[v]=parent[parent[v]];v=parent[v];}return v;};
    auto log_evidence=[&](int a,int b) {
        double maximum=-INFINITY;
        for(int g=0;g<k;++g)maximum=std::max(maximum,sums[a*k+g]+(b<0?0:sums[b*k+g]));
        double total=0.;for(int g=0;g<k;++g)total+=std::exp(sums[a*k+g]+(b<0?0:sums[b*k+g])-maximum);
        return maximum+std::log(total)-std::log(static_cast<double>(k));
    };
    for(int v=0;v<n;++v)evidence[v]=log_evidence(v,-1);
    // score, negative roots, versions: equal scores prefer lower root IDs.
    using Entry=std::tuple<double,int,int,int,int>;
    std::priority_queue<Entry> queue;
    auto offer=[&](int a,int b){if(a>b)std::swap(a,b);if(a==b)return;
        double gain=log_evidence(a,b)-evidence[a]-evidence[b];
        if(gain>1e-12)queue.emplace(gain,-a,-b,version[a],version[b]);};
    for(int y=0;y<h;++y)for(int x=0;x<w;++x){int a=y*w+x;if(!support[a])continue;
        if(x+1<w&&support[a+1]){neighbours[a].insert(a+1);neighbours[a+1].insert(a);offer(a,a+1);}
        if(y+1<h&&support[a+w]){neighbours[a].insert(a+w);neighbours[a+w].insert(a);offer(a,a+w);}}
    int merges=0;double total_gain=0.;
    while(!queue.empty()) {
        auto item=queue.top();queue.pop();int a=-std::get<1>(item),b=-std::get<2>(item);
        if(parent[a]!=a||parent[b]!=b||version[a]!=std::get<3>(item)||version[b]!=std::get<4>(item))continue;
        double gain=log_evidence(a,b)-evidence[a]-evidence[b];if(gain<=1e-12)continue;
        if(size[a]<size[b]||(size[a]==size[b]&&a>b))std::swap(a,b);
        parent[b]=a;size[a]+=size[b];++version[a];++version[b];++merges;total_gain+=gain;
        for(int g=0;g<k;++g)sums[a*k+g]+=sums[b*k+g];evidence[a]=log_evidence(a,-1);
        std::unordered_set<int> joined;
        for(int v:neighbours[a]){int r=root(v);if(r!=a)joined.insert(r);}
        for(int v:neighbours[b]){int r=root(v);if(r!=a)joined.insert(r);}
        neighbours[a].swap(joined);neighbours[b].clear();
        for(int v:neighbours[a]){neighbours[v].erase(b);neighbours[v].erase(a);neighbours[v].insert(a);offer(a,v);}
    }
    int count=0;
    for(int v=0;v<n;++v){int r=root(v);if(compact[r]<0)compact[r]=count++;output[v]=compact[r];}
    diagnostics[0]=merges;diagnostics[1]=count;diagnostics[2]=total_gain;
    return 0;
}

// Sorted-edge internal-contrast segmentation; no additional size merge pass.
extern "C" int bora_contrast_components(const int32_t* a,const int32_t* b,
        const double* cost,int32_t* output,int n,int m,double area,double scale) {
    if(n<1||m<0||area<=0||scale<=0)return 1;
    std::vector<int> parent(n),size(n,1),compact(n,-1);
    std::vector<double> internal(n,0.);
    for(int i=0;i<n;++i)parent[i]=i;
    auto root=[&](int v){while(parent[v]!=v){parent[v]=parent[parent[v]];v=parent[v];}return v;};
    for(int e=0;e<m;++e){
        if(a[e]<0||a[e]>=n||b[e]<0||b[e]>=n||!std::isfinite(cost[e])||cost[e]<0||
           (e>0&&cost[e]<cost[e-1]))return 2;
        int x=root(a[e]),y=root(b[e]);if(x==y)continue;
        double threshold=std::min(internal[x]+scale/(area*size[x]),internal[y]+scale/(area*size[y]));
        if(cost[e]<=threshold){
            if(size[x]<size[y]||(size[x]==size[y]&&x>y))std::swap(x,y);
            parent[y]=x;size[x]+=size[y];internal[x]=std::max(cost[e],std::max(internal[x],internal[y]));
        }
    }
    int count=0;
    for(int i=0;i<n;++i){int r=root(i);if(compact[r]<0)compact[r]=count++;output[i]=compact[r];}
    return 0;
}

static void cheb_simplex(std::vector<double>& q, std::vector<double>& ordered) {
    ordered=q; std::sort(ordered.begin(),ordered.end(),std::greater<double>());
    double sum=0,threshold=0;
    for(size_t j=0;j<q.size();++j) {
        sum+=ordered[j];
        if(ordered[j]>(sum-1)/(j+1)) threshold=(sum-1)/(j+1);
    }
    for(double& value:q) value=std::max(0.,value-threshold);
}

// Independent Chambolle-Pock solves. The dual of beta*||r||_infinity is the
// indicator of an L1 ball. The gap uses the exact simplex-prox conjugate.
extern "C" int bora_chebyshev_palette(const float* x,const float* prior,
        const double* colors,float* output,double* gaps,int32_t* used,
        int64_t n,int k,double ridge,double beta,double step,
        double tolerance,int iterations,int threads) {
    if(n<1||k<2||ridge<=0||beta<=0||step<=0||iterations<1||threads<1) return 1;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(int64_t row=0;row<n;++row) {
        std::vector<double> q(k),bar(k),changed(k),star(k),ordered(k);
        for(int g=0;g<k;++g) q[g]=prior[row*k+g];
        cheb_simplex(q,ordered);bar=q;
        double dual[3]={0,0,0},gap=std::numeric_limits<double>::infinity();
        int count=0;
        for(int it=0;it<iterations;++it) {
            double sumabs=0,absolute[3];
            for(int j=0;j<3;++j) {
                double residual=-x[row*3+j];
                for(int g=0;g<k;++g) residual+=bar[g]*colors[g*3+j];
                dual[j]+=step*residual;absolute[j]=std::abs(dual[j]);sumabs+=absolute[j];
            }
            if(sumabs>beta) {
                std::sort(absolute,absolute+3,std::greater<double>());
                double sum=0,threshold=0;
                for(int j=0;j<3;++j) {sum+=absolute[j];if(absolute[j]>(sum-beta)/(j+1))threshold=(sum-beta)/(j+1);}
                for(double& value:dual) value=std::copysign(std::max(0.,std::abs(value)-threshold),value);
            }
            for(int g=0;g<k;++g) {
                double linear=0;for(int j=0;j<3;++j)linear+=colors[g*3+j]*dual[j];
                changed[g]=(q[g]-step*linear+step*ridge*prior[row*k+g])/(1+step*ridge);
            }
            cheb_simplex(changed,ordered);
            for(int g=0;g<k;++g) {bar[g]=2*changed[g]-q[g];q[g]=changed[g];}
            count=it+1;
            if(count%10==0||count==iterations) {
                double primal=0,dualvalue=0,maxerror=0;
                for(int g=0;g<k;++g) {
                    double linear=0;for(int j=0;j<3;++j)linear+=colors[g*3+j]*dual[j];
                    star[g]=prior[row*k+g]-linear/ridge;
                }
                cheb_simplex(star,ordered);
                for(int g=0;g<k;++g) {
                    double linear=0;for(int j=0;j<3;++j)linear+=colors[g*3+j]*dual[j];
                    primal+=.5*ridge*std::pow(q[g]-prior[row*k+g],2);
                    dualvalue+=.5*ridge*std::pow(star[g]-prior[row*k+g],2)+star[g]*linear;
                }
                for(int j=0;j<3;++j) {
                    double error=-x[row*3+j];
                    for(int g=0;g<k;++g)error+=q[g]*colors[g*3+j];
                    maxerror=std::max(maxerror,std::abs(error));dualvalue-=x[row*3+j]*dual[j];
                }
                gap=std::max(0.,primal+beta*maxerror-dualvalue);
                if(gap<tolerance)break;
            }
        }
        gaps[row]=gap;used[row]=count;
        for(int g=0;g<k;++g)output[row*k+g]=float(q[g]);
    }
    return 0;
}

// Class-specific quadratic Gaussian scores. Each row is independent, so
// one/two-thread calculations have identical floating-point operation order.
extern "C" int bora_gaussian_logits(const float* x, const double* means,
        const double* precision, const double* intercept, float* output,
        int64_t n, int dimensions, int classes, int threads) {
    if(n<0 || dimensions<1 || classes<1 || threads<1) return 1;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(int64_t i=0;i<n;++i) {
        for(int g=0;g<classes;++g) {
            double quadratic=0;
            for(int a=0;a<dimensions;++a) {
                double row=0;
                for(int b=0;b<dimensions;++b)
                    row+=precision[(g*dimensions+a)*dimensions+b]
                        *(x[i*dimensions+b]-means[g*dimensions+b]);
                quadratic+=(x[i*dimensions+a]-means[g*dimensions+a])*row;
            }
            output[i*classes+g]=float(intercept[g]-.5*quadratic);
        }
    }
    return 0;
}

// Euclidean projection on a simplex whose designated coordinate is maximal.
// Star-order isotonic regression pools the root with violating rivals;
// thresholding that fitted vector gives the nonnegative unit-sum projection.
static void dominant_projection(const std::vector<double>& value, int own,
        std::vector<double>& out, std::vector<double>& ordered) {
    const int k=value.size(); ordered.clear();
    for(int j=0;j<k;++j) if(j!=own) ordered.push_back(value[j]);
    std::sort(ordered.begin(),ordered.end(),std::greater<double>());
    double total=value[own], top=total;
    for(int j=0;j<k-1;++j) {
        total+=ordered[j]; top=std::max(top,total/(j+2));
    }
    for(int j=0;j<k;++j) out[j]=(j==own?top:std::min(value[j],top));
    ordered=out;
    std::sort(ordered.begin(),ordered.end(),std::greater<double>());
    total=0; double threshold=0;
    for(int j=0;j<k;++j) {
        total+=ordered[j];
        if(ordered[j]>(total-1)/(j+1)) threshold=(total-1)/(j+1);
    }
    for(int j=0;j<k;++j) out[j]=std::max(0.,out[j]-threshold);
}

extern "C" int bora_dominance_qp(const float* x, const float* prior,
        const double* colors, const float* initial, const int32_t* labels,
        float* output, double* residual, int32_t* iterations,
        int64_t n, int k, double ridge, double tolerance, int limit, int threads) {
    if(n<1 || k<1 || ridge<=0 || tolerance<=0 || limit<1 || threads<1) return 1;
    for(int64_t i=0;i<n;++i) if(labels[i]<0 || labels[i]>=k) return 2;
    std::vector<double> h(k*k,0.);
    double lipschitz=0;
    for(int a=0;a<k;++a) {
        double row=0;
        for(int b=0;b<k;++b) {
            double value=(a==b?ridge:0.);
            for(int d=0;d<3;++d) value+=colors[3*a+d]*colors[3*b+d];
            h[a*k+b]=value; row+=std::abs(value);
        }
        lipschitz=std::max(lipschitz,row);
    }
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(int64_t i=0;i<n;++i) {
        std::vector<double> b(k),q(k),z(k),next(k),gradient(k),query(k),projected(k),ordered;
        ordered.reserve(k);
        for(int a=0;a<k;++a) {
            b[a]=ridge*prior[i*k+a];
            for(int d=0;d<3;++d) b[a]+=colors[3*a+d]*x[3*i+d];
            z[a]=initial[i*k+a];
        }
        dominant_projection(z,labels[i],q,ordered); z=q;
        auto energy=[&](const std::vector<double>& value) {
            double sum=0;
            for(int a=0;a<k;++a) {
                sum-=b[a]*value[a];
                for(int b0=0;b0<k;++b0) sum+=.5*value[a]*h[a*k+b0]*value[b0];
            }
            return sum;
        };
        auto grad=[&](const std::vector<double>& value) {
            for(int a=0;a<k;++a) {
                gradient[a]=-b[a];
                for(int b0=0;b0<k;++b0) gradient[a]+=h[a*k+b0]*value[b0];
            }
        };
        double momentum=1, value=energy(q), error=std::numeric_limits<double>::infinity();
        int used=0;
        for(int it=0;it<limit;++it) {
            grad(z);
            for(int a=0;a<k;++a) query[a]=z[a]-gradient[a]/lipschitz;
            dominant_projection(query,labels[i],next,ordered);
            double changed=energy(next);
            if(changed>value+1e-13) {
                momentum=1; grad(q);
                for(int a=0;a<k;++a) query[a]=q[a]-gradient[a]/lipschitz;
                dominant_projection(query,labels[i],next,ordered); changed=energy(next);
            }
            const double updated=(1+std::sqrt(1+4*momentum*momentum))/2;
            for(int a=0;a<k;++a) z[a]=next[a]+(momentum-1)/updated*(next[a]-q[a]);
            q=next; momentum=updated; value=changed; used=it+1;
            if(it%8==0 || it==limit-1) {
                grad(q);
                for(int a=0;a<k;++a) query[a]=q[a]-gradient[a];
                dominant_projection(query,labels[i],projected,ordered);
                error=0;
                for(int a=0;a<k;++a) error=std::max(error,std::abs(q[a]-projected[a]));
                if(error<tolerance) break;
            }
        }
        for(int a=0;a<k;++a) {output[i*k+a]=float(q[a]);q[a]=output[i*k+a];}
        grad(q);
        for(int a=0;a<k;++a) query[a]=q[a]-gradient[a];
        dominant_projection(query,labels[i],projected,ordered);
        error=0;
        for(int a=0;a<k;++a) error=std::max(error,std::abs(q[a]-projected[a]));
        residual[i]=error; iterations[i]=used;
    }
    return 0;
}

extern "C" int bora_local_mixture_logits(const float* x, const double* coefficient,
        const double* intercept, const float* mixing, float* output,
        int64_t n, int dimensions, int components, int threads) {
    if(n<0 || dimensions<1 || components<1 || threads<1) return 1;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(int64_t i=0;i<n;++i) {
        double maximum=-std::numeric_limits<double>::infinity();
        double sum=0;
        for(int g=0;g<components;++g) {
            double score=intercept[g]+std::log(std::max(double(mixing[i*components+g]),1e-30));
            for(int j=0;j<dimensions;++j) score+=x[i*dimensions+j]*coefficient[g*dimensions+j];
            if(score>maximum) {sum=sum*std::exp(maximum-score)+1;maximum=score;}
            else sum+=std::exp(score-maximum);
        }
        output[i]=float(maximum+std::log(sum));
    }
    return 0;
}

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

// Exact signed categorical count differences for each own/rival comparison.
// The product over rivals is a composite likelihood, not the joint K-class
// winner probability. Lower class indices win ties. Probability derivatives
// account for p(neutral)=1-p(own)-p(rival), before the Python softmax chain rule.
extern "C" int bora_categorical_winner(const double* p, const int32_t* labels,
    double* logp, double* gradient, int bags, int pixels, int classes, int threads) {
    if(bags<1 || pixels<1 || classes<2 || threads<1) return -1;
    for(int b=0;b<bags;++b) if(labels[b]<0 || labels[b]>=classes) return -2;
    const int width=2*pixels+1, zero=pixels;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(int b=0;b<bags;++b) {
        const int own=labels[b];
        double* grad=gradient+static_cast<int64_t>(b)*pixels*classes;
        std::fill(grad,grad+pixels*classes,0.);
        logp[b]=0.;
        for(int rival=0;rival<classes;++rival) if(rival!=own) {
            std::vector<double> forward((pixels+1)*width,0.),backward((pixels+1)*width,0.);
            forward[zero]=1.; backward[pixels*width+zero]=1.;
            for(int i=0;i<pixels;++i) {
                const double* row=p+(static_cast<int64_t>(b)*pixels+i)*classes;
                const double a=row[own],d=row[rival],n=std::max(0.,1.-a-d);
                for(int s=-i;s<=i;++s) {
                    const double v=forward[i*width+zero+s];
                    forward[(i+1)*width+zero+s]+=n*v;
                    forward[(i+1)*width+zero+s+1]+=a*v;
                    forward[(i+1)*width+zero+s-1]+=d*v;
                }
            }
            for(int i=pixels-1;i>=0;--i) {
                const double* row=p+(static_cast<int64_t>(b)*pixels+i)*classes;
                const double a=row[own],d=row[rival],n=std::max(0.,1.-a-d);
                const int length=pixels-i-1;
                for(int s=-length;s<=length;++s) {
                    const double v=backward[(i+1)*width+zero+s];
                    backward[i*width+zero+s]+=n*v;
                    backward[i*width+zero+s+1]+=a*v;
                    backward[i*width+zero+s-1]+=d*v;
                }
            }
            const int threshold=(rival<own?1:0);
            double probability=0.;
            for(int s=threshold;s<=pixels;++s) probability+=forward[pixels*width+zero+s];
            logp[b]+=std::log(std::max(probability,1e-300));
            if(probability<=1e-300) continue; // derivative of the explicit floor
            for(int i=0;i<pixels;++i) {
                double lower=0.,same=0.;
                const int rest=pixels-i-1;
                for(int s=-i;s<=i;++s) {
                    const double v=forward[i*width+zero+s];
                    const int a=threshold-1-s,d=threshold-s;
                    if(a>=-rest && a<=rest) lower+=v*backward[(i+1)*width+zero+a];
                    if(d>=-rest && d<=rest) same+=v*backward[(i+1)*width+zero+d];
                }
                grad[i*classes+own]-=lower/probability;
                grad[i*classes+rival]+=same/probability;
            }
        }
    }
    return 0;
}

// Signed depth of a point in a convex polytope whose facet normals have unit
// length. Stream facets rather than materializing a pixels-by-facets matrix.
extern "C" int bora_hull_depth(const float* x, const double* planes, float* out,
    int64_t n, int d, int facets, int threads) {
    if(n<0 || d<1 || facets<1 || threads<1) return -1;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(int64_t i=0;i<n;++i) {
        double maximum=-INFINITY;
        for(int f=0;f<facets;++f) {
            double v=planes[f*(d+1)+d];
            for(int j=0;j<d;++j) v+=planes[f*(d+1)+j]*x[i*d+j];
            maximum=std::max(maximum,v);
        }
        out[i]=static_cast<float>(-maximum);
    }
    return 0;
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

// General strictly convex simplex quadratic. New exports leave the original
// cycle-one numerical path unchanged. h_stride is either zero (shared H) or
// k*k (one observation-dependent Hessian per row).
static void quadratic_face(const double* h,const double* b,const float* prior,
    float* output,double& violation,int& used,int k,double tolerance,int max_iter) {
    std::vector<double> p(k,0),z(k,0);
    std::vector<int> active;
    int first=0;
    for(int c=1;c<k;++c) if(prior[c]>prior[first]) first=c;
    active.push_back(first);p[first]=1.;violation=INFINITY;used=0;
    for(;used<max_iter;++used) {
        int s=static_cast<int>(active.size()),m=s+1;
        std::vector<double> a(m*m,0),rhs(m,0),answer;
        for(int i=0;i<s;++i) {
            rhs[i]=b[active[i]];
            for(int j=0;j<s;++j) a[i*m+j]=h[active[i]*k+active[j]];
            a[i*m+s]=1;a[s*m+i]=1;
        }
        rhs[s]=1;
        if(!solve(a,rhs,answer,m)) break;
        std::fill(z.begin(),z.end(),0.);
        for(int i=0;i<s;++i) z[active[i]]=answer[i];
        double alpha=1.;
        for(int c:active) if(z[c]<0.)
            alpha=std::min(alpha,p[c]/std::max(p[c]-z[c],1e-30));
        for(int c=0;c<k;++c) p[c]+=alpha*(z[c]-p[c]);
        if(alpha<1.-1e-12) {
            active.erase(std::remove_if(active.begin(),active.end(),
                [&](int c){if(p[c]<1e-11){p[c]=0.;return true;}return false;}),active.end());
            continue;
        }
        int entering=-1;double smallest=0.;violation=0.;
        for(int c=0;c<k;++c) {
            double gradient=-b[c]+answer[s];
            for(int j=0;j<k;++j) gradient+=h[c*k+j]*p[j];
            bool present=std::find(active.begin(),active.end(),c)!=active.end();
            if(present) violation=std::max(violation,std::abs(gradient));
            else if(gradient<smallest){smallest=gradient;entering=c;}
        }
        violation=std::max(violation,-smallest);
        if(violation<=tolerance){++used;break;}
        if(entering<0) break;
        active.push_back(entering);
    }
    double mass=0.;
    for(double& value:p){value=std::max(0.,value);mass+=value;}
    for(int c=0;c<k;++c) output[c]=static_cast<float>(p[c]/mass);
}

extern "C" int bora_quadratic_simplex(const double* h,const double* b,
    const float* prior,float* out,double* residual,int* iterations,
    int64_t n,int k,int h_stride,double tolerance,int max_iter,int threads) {
    if(n<1||k<1||(h_stride!=0&&h_stride!=k*k)||tolerance<=0||threads<1) return -1;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(int64_t row=0;row<n;++row)
        quadratic_face(h+row*h_stride,b+row*k,prior+row*k,out+row*k,
            residual[row],iterations[row],k,tolerance,max_iter);
    return 0;
}

// Exact minimization over the union of every one- and two-class face.
extern "C" int bora_pair_palette(const float* rgb,const float* prior,
    const double* colors,float* out,int64_t n,int k,double ridge,int threads) {
    if(n<1||k<1||ridge<=0||threads<1) return -1;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(int64_t row=0;row<n;++row) {
        double best=INFINITY,best_t=1.;int ai=0,bi=0;
        for(int a=0;a<k;++a) for(int b=a;b<k;++b) {
            double dot=0.,norm=0.;
            for(int j=0;j<3;++j) {
                double d=colors[a*3+j]-colors[b*3+j];
                dot+=d*(rgb[row*3+j]-colors[b*3+j]);norm+=d*d;
            }
            double t=a==b?1.:(dot+ridge*(1.+prior[row*k+a]-prior[row*k+b]))/(norm+2.*ridge);
            t=std::max(0.,std::min(1.,t));double cost=0.;
            for(int j=0;j<3;++j) {
                double d=t*colors[a*3+j]+(1.-t)*colors[b*3+j]-rgb[row*3+j];cost+=d*d;
            }
            for(int c=0;c<k;++c) {
                double q=c==a?t:(c==b?1.-t:0.);
                double d=q-prior[row*k+c];cost+=ridge*d*d;
            }
            if(cost<best){best=cost;best_t=t;ai=a;bi=b;}
        }
        for(int c=0;c<k;++c) out[row*k+c]=0.f;
        out[row*k+ai]=static_cast<float>(best_t);
        if(ai!=bi) out[row*k+bi]=static_cast<float>(1.-best_t);
    }
    return 0;
}

static double entropy_objective(const std::vector<double>& p,const float* x,
    const float* prior,const double* colors,int k,double ridge) {
    double value=0.;
    for(int j=0;j<3;++j) {
        double d=-x[j];for(int c=0;c<k;++c)d+=p[c]*colors[3*c+j];value+=.5*d*d;
    }
    for(int c=0;c<k;++c)
        value+=ridge*p[c]*std::log(std::max(p[c],1e-300)/std::max(double(prior[c]),1e-12));
    return value;
}

extern "C" int bora_entropy_palette(const float* rgb,const float* prior,
    const double* colors,float* out,double* residual,int* iterations,
    int64_t n,int k,double ridge,double tolerance,int max_iter,int threads) {
    if(n<1||k<1||ridge<=0||threads<1) return -1;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(int64_t row=0;row<n;++row) {
        const float *x=rgb+3*row,*pi=prior+k*row;
        std::vector<double> p(k),candidate(k),g(k);double mass=0.;
        for(int c=0;c<k;++c){p[c]=std::max(double(pi[c]),1e-8);mass+=p[c];}
        for(double& v:p)v/=mass;
        int used=0;double violation=INFINITY;
        for(;used<max_iter;++used) {
            double error[3]={-x[0],-x[1],-x[2]};
            for(int c=0;c<k;++c) for(int j=0;j<3;++j)error[j]+=p[c]*colors[c*3+j];
            double mean_g=0.;
            for(int c=0;c<k;++c) {
                g[c]=ridge*(std::log(p[c]/std::max(double(pi[c]),1e-12))+1.);
                for(int j=0;j<3;++j)g[c]+=colors[3*c+j]*error[j];mean_g+=g[c]/k;
            }
            violation=0.;for(int c=0;c<k;++c)violation=std::max(violation,std::abs(g[c]-mean_g));
            if(violation<tolerance)break;
            int m=k+1;std::vector<double> a(m*m,0.),rhs(m,0.),answer;
            for(int c=0;c<k;++c) {
                rhs[c]=-g[c];a[c*m+k]=1.;a[k*m+c]=1.;
                for(int d=0;d<k;++d) {
                    for(int j=0;j<3;++j)a[c*m+d]+=colors[3*c+j]*colors[3*d+j];
                    if(c==d)a[c*m+d]+=ridge/p[c];
                }
            }
            if(!solve(a,rhs,answer,m))break;
            double alpha=1.,descent=0.;
            for(int c=0;c<k;++c) {
                if(answer[c]<0.)alpha=std::min(alpha,-.99*p[c]/answer[c]);
                descent+=g[c]*answer[c];
            }
            double before=entropy_objective(p,x,pi,colors,k,ridge);
            bool accepted=false;
            for(int backtrack=0;backtrack<35;++backtrack) {
                for(int c=0;c<k;++c)candidate[c]=p[c]+alpha*answer[c];
                double after=entropy_objective(candidate,x,pi,colors,k,ridge);
                if(after<=before+1e-4*alpha*descent+1e-14){accepted=true;break;}
                alpha*=.5;
            }
            if(!accepted)break;
            p.swap(candidate);
        }
        for(int c=0;c<k;++c)out[row*k+c]=static_cast<float>(p[c]);
        residual[row]=violation;iterations[row]=used;
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

// One native pixel pass for semantic confusion and reference-component recall.
// Reference-zero pixels never enter the confusion matrix or correctness counts.
extern "C" int bora_region_counts(const uint8_t* truth,const uint8_t* prediction,
    const int32_t* components,int64_t* confusion,int64_t* correct,
    int64_t pixels,int component_count,int threads) {
    if(pixels<0||component_count<1||threads<1) return -1;
    std::fill(confusion,confusion+256*256,0);
    std::fill(correct,correct+component_count,0);
    #pragma omp parallel num_threads(threads)
    {
        std::vector<int64_t> local_confusion(256*256,0),local_correct(component_count,0);
        #pragma omp for schedule(static)
        for(int64_t i=0;i<pixels;++i) if(truth[i]>0) {
            ++local_confusion[static_cast<int>(truth[i])*256+prediction[i]];
            if(truth[i]==prediction[i]) ++local_correct[components[i]];
        }
        #pragma omp critical
        {
            for(int i=0;i<256*256;++i) confusion[i]+=local_confusion[i];
            for(int i=0;i<component_count;++i) correct[i]+=local_correct[i];
        }
    }
    return 0;
}

// Exact categorical diversity at selected white pixels for the supplied
// discrete ellipse. A cheap rectangular min/max prefilter is done in Python.
extern "C" int bora_white_diversity(const uint16_t* labels,const int64_t* points,
    const int32_t* offsets,uint8_t* result,int height,int width,
    int64_t point_count,int offset_count,int threads) {
    if(height<1||width<1||point_count<0||offset_count<1||threads<1) return -1;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(int64_t i=0;i<point_count;++i) {
        int y=points[i]/width,x=points[i]%width,first=0;
        result[i]=0;
        for(int j=0;j<offset_count;++j) {
            int yy=y+offsets[j*2],xx=x+offsets[j*2+1];
            if(yy<0||xx<0||yy>=height||xx>=width) continue;
            int label=labels[static_cast<int64_t>(yy)*width+xx];
            if(label>0) {
                if(first==0) first=label;
                else if(label!=first){result[i]=1;break;}
            }
        }
    }
    return 0;
}

// Exact Potts dynamic program on a supplied rooted tree, O(nodes*classes).
extern "C" int bora_tree_potts(const int32_t* parent,const int32_t* order,
    const double* unary,const double* penalty,int32_t* labels,double* energy,int n,int k) {
    if(n<1||k<1) return -1;
    std::vector<double> dp(unary,unary+static_cast<int64_t>(n)*k);
    for(int t=n-1;t>0;--t) {
        int v=order[t],p=parent[v];
        if(v<0||v>=n||p<0||p>=n||penalty[v]<0) return -2;
        double least=*std::min_element(&dp[v*k],&dp[(v+1)*k]);
        for(int c=0;c<k;++c) dp[p*k+c]+=std::min(dp[v*k+c],least+penalty[v]);
    }
    int root=order[0];
    labels[root]=std::min_element(&dp[root*k],&dp[(root+1)*k])-&dp[root*k];
    energy[0]=dp[root*k+labels[root]];
    for(int t=1;t<n;++t) {
        int v=order[t],p=parent[v],best=0;double score=INFINITY;
        for(int c=0;c<k;++c) {
            double value=dp[v*k+c]+(c==labels[p]?0:penalty[v]);
            if(value<score){score=value;best=c;}
        }
        labels[v]=best;
    }
    double actual=0;
    for(int v=0;v<n;++v) actual+=unary[v*k+labels[v]];
    for(int t=1;t<n;++t) {
        int v=order[t];if(labels[v]!=labels[parent[v]])actual+=penalty[v];
    }
    energy[1]=actual;return 0;
}

// Kruskal forest under fixed competing seed identities. Sorted edge order is
// part of the input; ties are resolved by the caller's lexicographic edge IDs.
extern "C" int bora_seeded_forest(const int32_t* a,const int32_t* b,
    const int32_t* seeds,int32_t* labels,int n,int edges) {
    if(n<1||edges<0)return -1;
    std::vector<int> parent(n),size(n,1),seed(seeds,seeds+n);
    for(int i=0;i<n;++i)parent[i]=i;
    auto find=[&](int x){while(parent[x]!=x){parent[x]=parent[parent[x]];x=parent[x];}return x;};
    for(int e=0;e<edges;++e) {
        if(a[e]<0||a[e]>=n||b[e]<0||b[e]>=n)return -2;
        int x=find(a[e]),y=find(b[e]);if(x==y)continue;
        if(seed[x]&&seed[y]&&seed[x]!=seed[y])continue;
        if(size[x]<size[y])std::swap(x,y);
        parent[y]=x;size[x]+=size[y];seed[x]=std::max(seed[x],seed[y]);
    }
    for(int i=0;i<n;++i)labels[i]=seed[find(i)];return 0;
}

// Iterative Tarjan decomposition avoids recursion depth on long white paths.
// White border pixels join one exterior root. Only nontrivial root blocks give
// two independent escape routes; border pixels themselves are always exterior.
extern "C" int bora_biconnected_white(const uint8_t* white,uint8_t* result,int h,int w) {
    if(h<1||w<1)return -1;
    int n=h*w,root=n;
    std::vector<std::vector<int>> adj(n+1);
    std::fill(result,result+n,0);
    auto edge=[&](int a,int b){adj[a].push_back(b);adj[b].push_back(a);};
    for(int y=0;y<h;++y)for(int x=0;x<w;++x) {
        int v=y*w+x;if(!white[v])continue;
        if(x+1<w&&white[v+1])edge(v,v+1);
        if(y+1<h&&white[v+w])edge(v,v+w);
        if(y==0||y==h-1||x==0||x==w-1){edge(root,v);result[v]=1;}
    }
    std::vector<int> disc(n+1,0),low(n+1,0),parent(n+1,-1),next(n+1,0),stack(1,root);
    std::vector<std::pair<int,int>> edges;
    int clock=1;disc[root]=low[root]=clock;
    while(!stack.empty()) {
        int v=stack.back();
        if(next[v]<static_cast<int>(adj[v].size())) {
            int u=adj[v][next[v]++];
            if(!disc[u]) {
                parent[u]=v;edges.emplace_back(v,u);disc[u]=low[u]=++clock;stack.push_back(u);
            } else if(u!=parent[v]&&disc[u]<disc[v]) {
                low[v]=std::min(low[v],disc[u]);edges.emplace_back(v,u);
            }
        } else {
            stack.pop_back();int p=parent[v];if(p<0)continue;
            low[p]=std::min(low[p],low[v]);
            if(low[v]>=disc[p]) {
                std::vector<int> block;bool has_root=false;
                while(!edges.empty()) {
                    auto e=edges.back();edges.pop_back();
                    block.push_back(e.first);block.push_back(e.second);
                    has_root|=e.first==root||e.second==root;
                    if(e.first==p&&e.second==v)break;
                }
                // At least three vertices are required; a bridge is not redundant.
                if(has_root&&block.size()>=6)for(int u:block)if(u<n)result[u]=1;
            }
        }
    }
    return 0;
}
