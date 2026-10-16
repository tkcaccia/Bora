// Fast triangular-mesh regularization for categorical internal boundaries.
// Build: g++ -O3 -std=c++17 -fopenmp -shared -fPIC tri_mesh_refine.cpp -o libbora_tri_mesh.so
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
struct Vertex {
    std::vector<float> occupancy;
    std::vector<float> occupancy_outer;
    std::vector<float> red, green, blue;
    std::vector<float> color_reliability;
    std::vector<uint8_t> has_color;
    explicit Vertex(int classes) : occupancy(classes, 0),
        occupancy_outer(classes, 0), red(classes, 0),
        green(classes, 0), blue(classes, 0),
        color_reliability(classes, 0), has_color(classes, 0) {}
};

inline uint32_t window_sum(const std::vector<uint32_t>& integral, int stride,
                           int x0, int y0, int x1, int y1) {
    const size_t a = static_cast<size_t>(y0) * stride + x0;
    const size_t b = static_cast<size_t>(y0) * stride + x1;
    const size_t c = static_cast<size_t>(y1) * stride + x0;
    const size_t d = static_cast<size_t>(y1) * stride + x1;
    return integral[d] - integral[b] - integral[c] + integral[a];
}
}

// Returns number of changed pixels, or a negative error code. RGB is optional;
// its rows are packed width*3 bytes. Callers own all input and output storage.
extern "C" int64_t bora_tri_mesh_u16(
    const uint16_t* source, const uint8_t* rgb, uint16_t* output,
    int width, int height, int mesh_spacing, int support_radius,
    int band_radius, float color_weight, int requested_threads,
    int post_smooth_radius) {
    if (!source || !output || width < 2 || height < 2 || mesh_spacing < 2 ||
        support_radius < 1 || band_radius < 1 || post_smooth_radius < 0 ||
        color_weight < 0 ||
        color_weight > 1 || (color_weight > 0 && !rgb)) return -1;
    const size_t pixels = static_cast<size_t>(width) * height;
    if (pixels > static_cast<size_t>(std::numeric_limits<int>::max())) return -2;
    std::copy(source, source + pixels, output);

    std::vector<uint16_t> classes;
    std::vector<int16_t> lookup(65536, -1);
    for (size_t i = 0; i < pixels; ++i) {
        uint16_t label = source[i];
        if (label && lookup[label] == -1) {
            if (classes.size() >= 255) return -3;
            lookup[label] = static_cast<int16_t>(classes.size());
            classes.push_back(label);
        }
    }
    if (classes.size() < 2) return 0;
    const int n = static_cast<int>(classes.size());

    // Four-neighbour distance to a tissue-to-tissue edge. External tissue
    // contours are deliberately fixed by this stage.
    std::vector<uint16_t> distance(pixels, 65535);
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
        size_t i = static_cast<size_t>(y) * width + x;
        uint16_t a = source[i];
        if (!a) continue;
        bool edge = (x && source[i-1] && source[i-1] != a) ||
                    (x+1 < width && source[i+1] && source[i+1] != a) ||
                    (y && source[i-width] && source[i-width] != a) ||
                    (y+1 < height && source[i+width] && source[i+width] != a);
        if (edge) distance[i] = 0;
    }
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
        size_t i = static_cast<size_t>(y) * width + x;
        if (x) distance[i] = std::min<int>(distance[i], distance[i-1]+1);
        if (y) distance[i] = std::min<int>(distance[i], distance[i-width]+1);
    }
    for (int y = height-1; y >= 0; --y) for (int x = width-1; x >= 0; --x) {
        size_t i = static_cast<size_t>(y) * width + x;
        if (x+1 < width) distance[i] = std::min<int>(distance[i], distance[i+1]+1);
        if (y+1 < height) distance[i] = std::min<int>(distance[i], distance[i+width]+1);
    }

    // Summed-area tables make support samples at mesh vertices independent of
    // the smoothing radius. One label table is built at a time to bound memory.
    const int integral_stride = width + 1;
    std::vector<uint32_t> integral(static_cast<size_t>(width+1)*(height+1));
    std::vector<int> xs, ys;
    for (int x = 0; x < width-1; x += mesh_spacing) xs.push_back(x);
    xs.push_back(width-1);
    for (int y = 0; y < height-1; y += mesh_spacing) ys.push_back(y);
    ys.push_back(height-1);
    const int nx = static_cast<int>(xs.size()), ny = static_cast<int>(ys.size());
    std::vector<Vertex> vertices;
    vertices.reserve(static_cast<size_t>(nx)*ny);
    for (int i = 0; i < nx*ny; ++i) vertices.emplace_back(n);

    for (int label_index = 0; label_index < n; ++label_index) {
        uint16_t label = classes[label_index];
        std::fill(integral.begin(), integral.end(), 0);
        for (int y = 0; y < height; ++y) {
            uint32_t row_sum = 0;
            for (int x = 0; x < width; ++x) {
                row_sum += source[static_cast<size_t>(y)*width+x] == label;
                integral[static_cast<size_t>(y+1)*integral_stride+x+1] =
                    integral[static_cast<size_t>(y)*integral_stride+x+1] + row_sum;
            }
        }
        for (int gy = 0; gy < ny; ++gy) for (int gx = 0; gx < nx; ++gx) {
            const int cx = xs[gx], cy = ys[gy];
            const int x0 = std::max(0, cx-support_radius);
            const int y0 = std::max(0, cy-support_radius);
            const int x1 = std::min(width, cx+support_radius+1);
            const int y1 = std::min(height, cy+support_radius+1);
            auto& vertex = vertices[static_cast<size_t>(gy)*nx+gx];
            vertex.occupancy[label_index] = static_cast<float>(
                window_sum(integral, integral_stride, x0, y0, x1, y1)) /
                static_cast<float>((x1-x0)*(y1-y0));
            if (color_weight && rgb) {
                int outer_radius = std::max(support_radius+16,
                                            static_cast<int>(std::round(1.3f*support_radius)));
                int ox0 = std::max(0, cx-outer_radius);
                int oy0 = std::max(0, cy-outer_radius);
                int ox1 = std::min(width, cx+outer_radius+1);
                int oy1 = std::min(height, cy+outer_radius+1);
                vertex.occupancy_outer[label_index] = static_cast<float>(
                    window_sum(integral, integral_stride, ox0, oy0, ox1, oy1)) /
                    static_cast<float>((ox1-ox0)*(oy1-oy0));
                uint64_t r = 0, g = 0, b = 0, count = 0;
                for (int y = y0; y < y1; y += 8) for (int x = x0; x < x1; x += 8) {
                    size_t i = static_cast<size_t>(y)*width+x;
                    if (source[i] != label || distance[i] < 8) continue;
                    r += rgb[3*i]; g += rgb[3*i+1]; b += rgb[3*i+2]; ++count;
                }
                if (count) {
                    vertex.red[label_index] = static_cast<float>(r)/count;
                    vertex.green[label_index] = static_cast<float>(g)/count;
                    vertex.blue[label_index] = static_cast<float>(b)/count;
                    vertex.has_color[label_index] = 1;
                }
            }
        }
    }
    if (color_weight) {
        // Trust image color only where neighboring tissue prototypes are
        // distinguishable. This is evaluated locally at each mesh vertex.
        for (auto& vertex : vertices) for (int k = 0; k < n; ++k) {
            if (!vertex.has_color[k]) continue;
            float nearest = 1e9f;
            for (int other = 0; other < n; ++other) {
                if (other == k || !vertex.has_color[other]) continue;
                float dr = vertex.red[k]-vertex.red[other];
                float dg = vertex.green[k]-vertex.green[other];
                float db = vertex.blue[k]-vertex.blue[other];
                nearest = std::min(nearest,
                                   std::sqrt((dr*dr+dg*dg+db*db)/3.f));
            }
            if (nearest < 1e8f)
                vertex.color_reliability[k] =
                    std::clamp((nearest-20.f)/40.f, 0.f, 1.f);
        }
        for (auto& vertex : vertices) {
            float contrast = *std::max_element(vertex.color_reliability.begin(),
                                               vertex.color_reliability.end());
            for (int k = 0; k < n; ++k)
                vertex.occupancy[k] = contrast*vertex.occupancy[k] +
                                      (1.f-contrast)*vertex.occupancy_outer[k];
        }
    }

    // Each cell is split into two triangles. Barycentric interpolation turns
    // a coarse, stair-stepped mask into piecewise-linear class interfaces.
    int64_t changed = 0;
    #ifdef _OPENMP
    if (requested_threads > 0) omp_set_num_threads(requested_threads);
    #pragma omp parallel for reduction(+:changed) schedule(static)
    #endif
    for (int y = 0; y < height; ++y) {
        auto yi = std::upper_bound(ys.begin(), ys.end(), y);
        int gy = std::max(0, std::min(ny-2, static_cast<int>(yi-ys.begin())-1));
        float ty = static_cast<float>(y-ys[gy])/(ys[gy+1]-ys[gy]);
        for (int x = 0; x < width; ++x) {
            size_t i = static_cast<size_t>(y)*width+x;
            if (!source[i] || distance[i] > band_radius) continue;
            auto xi = std::upper_bound(xs.begin(), xs.end(), x);
            int gx = std::max(0, std::min(nx-2, static_cast<int>(xi-xs.begin())-1));
            float tx = static_cast<float>(x-xs[gx])/(xs[gx+1]-xs[gx]);
            const Vertex *a, *b, *c;
            float wa, wb, wc;
            if (tx >= ty) {
                a = &vertices[static_cast<size_t>(gy)*nx+gx];
                b = &vertices[static_cast<size_t>(gy)*nx+gx+1];
                c = &vertices[static_cast<size_t>(gy+1)*nx+gx+1];
                wa = 1-tx; wb = tx-ty; wc = ty;
            } else {
                a = &vertices[static_cast<size_t>(gy)*nx+gx];
                b = &vertices[static_cast<size_t>(gy+1)*nx+gx];
                c = &vertices[static_cast<size_t>(gy+1)*nx+gx+1];
                wa = 1-ty; wb = ty-tx; wc = tx;
            }
            float best_score = -1e9f;
            uint16_t best_label = source[i];
            for (int k = 0; k < n; ++k) {
                float score = wa*a->occupancy[k]+wb*b->occupancy[k]+wc*c->occupancy[k];
                if (color_weight && rgb) {
                    float ar = 0, ag = 0, ab = 0, weight = 0;
                    for (auto [v, w] : {std::pair{a,wa}, std::pair{b,wb}, std::pair{c,wc}}) {
                        if (v->has_color[k]) {
                            ar += w*v->red[k]; ag += w*v->green[k];
                            ab += w*v->blue[k]; weight += w;
                        }
                    }
                    if (weight > 0) {
                        float dr = rgb[3*i]-ar/weight;
                        float dg = rgb[3*i+1]-ag/weight;
                        float db = rgb[3*i+2]-ab/weight;
                        // A local stain difference of roughly 60 RGB levels
                        // is already meaningful in H&E. Scaling by the full
                        // 255 range made image guidance effectively inert.
                        float color_cost = (dr*dr+dg*dg+db*db)/(3*60.f*60.f);
                        float reliability = wa*a->color_reliability[k] +
                                            wb*b->color_reliability[k] +
                                            wc*c->color_reliability[k];
                        score -= color_weight*reliability*std::min(1.f, color_cost);
                    }
                }
                if (score > best_score) { best_score = score; best_label = classes[k]; }
            }
            output[i] = best_label;
            changed += best_label != source[i];
        }
    }
    if (post_smooth_radius > 0) {
        // A small symmetric vote rounds the piecewise-linear mesh corners.
        // Its input is the triangulated mask; original far interiors and all
        // tissue/background pixels remain fixed. This is deliberately after
        // triangulation rather than another blur of the coarse tile shape.
        std::vector<uint16_t> mesh_output(output, output + pixels);
        std::vector<uint16_t> rounded(mesh_output);
        std::vector<float> best(pixels, -1.f);
        for (uint16_t label : classes) {
            std::fill(integral.begin(), integral.end(), 0);
            for (int y = 0; y < height; ++y) {
                uint32_t row_sum = 0;
                for (int x = 0; x < width; ++x) {
                    row_sum += mesh_output[static_cast<size_t>(y)*width+x] == label;
                    integral[static_cast<size_t>(y+1)*integral_stride+x+1] =
                        integral[static_cast<size_t>(y)*integral_stride+x+1] + row_sum;
                }
            }
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (int y = 0; y < height; ++y) {
                int y0 = std::max(0, y-post_smooth_radius);
                int y1 = std::min(height, y+post_smooth_radius+1);
                for (int x = 0; x < width; ++x) {
                    size_t i = static_cast<size_t>(y)*width+x;
                    if (!source[i] || distance[i] > band_radius) continue;
                    int x0 = std::max(0, x-post_smooth_radius);
                    int x1 = std::min(width, x+post_smooth_radius+1);
                    float score = static_cast<float>(window_sum(
                        integral, integral_stride, x0, y0, x1, y1));
                    if (score > best[i]) { best[i] = score; rounded[i] = label; }
                }
            }
        }
        changed = 0;
        #ifdef _OPENMP
        #pragma omp parallel for reduction(+:changed) schedule(static)
        #endif
        for (size_t i = 0; i < pixels; ++i) {
            output[i] = rounded[i];
            changed += output[i] != source[i];
        }
    }
    return changed;
}
