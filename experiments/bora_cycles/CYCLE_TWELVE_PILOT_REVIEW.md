# Cycle12 engineering pilot, version1

This is not a completed cycle or a five-method selection. All24 calibrated
fields and all23 outputs were actually inspected as overview sheets; eight
targeted native label sheets were also inspected. The overview tool displayed
1876×1344 sheets from2304×1650 sources; native claims refer to original-detail
label sheets only, without claiming inspection of the exported overlays.

Prediction source: be54524. Report source: e9cd840. All552 native masks,120
retained-parent native comparisons and120 retained-parent accuracy comparisons
pass. Original111 archived preflight checks passed. Fine-reference scoring
started after the whole prediction freeze. All images are reused development
data; these differences are not independent validation or significance tests.

The release decision is **repair and repeat the engineering pilot**. Q12's
zero evidence in an unsupported4µm graph cell created an artificial smallest-ID
class fringe after fine-grid lifting. A failing independent regression
reproduced it. The repair preserves the declared original-parent fallback.
A C++ hash-set port targets Q15's queue-neighbour overhead without changing
the model. Both require a fresh frozen pilot before the full469-case run.
No candidate is selected or rejected by this pilot.

|Method|Mean IoU|Mean Bora seconds|Over1s|Adjusted utility|Actual observed limitation/benefit|
|---|---:|---:|---:|---:|---|
|B23|0.711820|0.441|0/24|0.711820|Diagnostic control, outside selection. B23 has the highest pilot overlap while filling obvious glass; control superiority remains disclosed.|
|Q03|0.709385|0.454|0/24|0.709385|Selectable pilot utility leader but obvious glass, grid and staircase support artifacts; high overlap is not valid foreground evidence.|
|P11|0.699589|0.473|1/24|0.699503|Native support fills real glass in CAM06/PUMA11/PESO26/Glea40; useful pale tissue coverage elsewhere.|
|Q02|0.692638|0.459|2/24|0.692389|More restrained enclosure in some glass edges; gray glass remains protected and open fat may be cut.|
|Q04|0.686768|0.369|0/24|0.686768|Often close to N08; removes false cyan and modestly changes class widths without solving missed microregions.|
|N08|0.686210|0.376|0/24|0.686210|Removes false cyan in IGNITE33 and changes fine island widths, but misses many small true regions.|
|Q05|0.685120|0.477|0/24|0.685120|Asymmetric fields remove false cyan locally; most whole masks remain near N08.|
|Q13|0.684741|0.216|0/24|0.684741|Close to fast palette masks; no clear visible tile-constraint breakthrough in this pilot.|
|Q01|0.702817|1.062|8/24|0.683530|Best of these support experiments at broad CAM06 gray glass, but residual speckles, other glass leaks and runtime cost persist.|
|Q10|0.683356|0.169|0/24|0.683356|Fast, generally close to J01; contour transport does not solve class ambiguity.|
|C17|0.683200|0.124|0/24|0.683200|Fast and avoids the IGNITE33 false cyan; still misses fine classes and flat-RGB islands.|
|B22|0.683076|0.140|0/24|0.683076|Diagnostic control, outside selection. B23 has the highest pilot overlap while filling obvious glass; control superiority remains disclosed.|
|Q14|0.682666|0.217|0/24|0.682666|Close to C17; six finite160-step solves miss stationarity, max projected residual0.00156097.|
|Q08|0.682316|0.405|0/24|0.682316|Sometimes creates holes/split islands; finite-probability ties use explicit parent tie-breaking.|
|J01|0.682282|0.162|0/24|0.682282|Fast reference; retains shared missed septa, false cyan and support limitations.|
|M04|0.682186|0.187|0/24|0.682186|Near J01 in appearance; uncertainty adds little visible benefit in this pilot.|
|Q07|0.681260|0.224|0/24|0.681260|Mostly parent-like and does not restore absent-coarse or flat-RGB regions.|
|Q12|0.681109|0.187|0/24|0.681109|Connectivity removes both false and true islands; native review exposed a zero-cell tie fringe requiring repair before full use.|
|Q09|0.682286|0.803|5/24|0.678283|Nearly the parent mask for extra matrix/probe cost; no compelling fine-region gain.|
|B21|0.675309|0.083|0/24|0.675309|Diagnostic control, outside selection. B23 has the highest pilot overlap while filling obvious glass; control superiority remains disclosed.|
|Q06|0.675134|0.525|2/24|0.674697|Fine stippling and lost thin PESO rings; occasional removal of false islands does not establish better anatomy.|
|Q11|0.674006|0.283|0/24|0.674006|Major simplification can remove false lobes, but also loses true islands and distorts compartments.|
|Q15|0.682165|5.098|21/24|0.474753|Version1 remains blocky and slow, with no proportional gain; hash-set implementation must prove output parity and measured speed.|

Times exclude cached Virchow2/RGB conversion, loading and rendering. Utility
is mean_i[IoU_i−0.05 max(seconds_i−1,0)], not an accuracy percentage.

The small-region atlas has769 components≤400µm², including66 absent-coarse
components. No method reaches50% class coverage for619 components, all methods
fully miss540, and14 components recovered by the coarse mask fall below50%
for every refinement. The common-support bound excludes50% reach for31; it is
not a bound on the separately owned native supports.

Q03's pilot utility lead is accompanied by glass filling in CAM06, false
protrusions in BEETLE01 and staircase blankets in GleasonXAI40. Q01 improves
broad gray-glass removal in CAM06 but does not solve the PUMA11 cleft and
PESO26 debris expansion. Internal gland clefts in BCSS18 remain filled.
TIGER71 and GleasonXAI40 show broad positive annotation polygons over glass;
removal remains counted as error under the unchanged scoring contract.

IGNITE33 supplies a different failure: retained J01/M04/support variants
create false cyan within a reference-green neck, while N08/Q04/Q05 and the
palette family avoid it. BCSS75's native crop has flat green converted RGB
where reference red microregions occur; every current method misses them.
The existing rarity diagnostics remain class-agnostic proposals, not new
expert biological identities. Cycles12–20 and the final full-five-conversion
comparison remain outstanding.
