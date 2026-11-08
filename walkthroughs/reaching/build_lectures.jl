# =============================================================================
# build_lectures.jl — run the reaching lectures headlessly in Pluto.
#
#   julia --project=. walkthroughs/reaching/build_lectures.jl            # all five
#   julia --project=. walkthroughs/reaching/build_lectures.jl 3 4        # just some
#
# Running a notebook fills any missing cache in walkthroughs/reaching/cache/ (every
# expensive result goes through `cached(...)`), reports cells that errored, and writes
# a static, self-contained HTML copy to walkthroughs/reaching/html/ — figures and the
# JavaScript animations work there without Julia; sliders need a live Pluto session.
# To recompute a result from scratch, delete its file in cache/ and rebuild.
# =============================================================================
using Pluto

const DIR = @__DIR__
const LECTURES = sort(filter(f -> occursin(r"^(0\d|appendix)_.*\.jl$", f), readdir(DIR)))

which = isempty(ARGS) ? LECTURES : [l for l in LECTURES if any(a -> startswith(l, lpad(a, 2, '0')), ARGS)]
session = Pluto.ServerSession()
session.options.server.disable_writing_notebook_files = true   # never rewrite the .jl files
nbad = 0
for f in which
    path = joinpath(DIR, f)
    println("▶ ", f); flush(stdout)
    t = @elapsed nb = Pluto.SessionActions.open(session, path; run_async=false)
    bad = filter(c -> c.errored, nb.cells)
    for c in bad
        global nbad += 1
        body = c.output.body
        msg = body isa Dict ? get(body, :msg, body) : body
        println("  ✗ cell ", c.cell_id, ": ", first(string(msg), 600))
        println("    code: ", first(replace(c.code, '\n' => " ⏎ "), 200))
    end
    out = joinpath(DIR, "html", replace(f, ".jl" => ".html"))
    mkpath(dirname(out))
    write(out, Pluto.generate_html(nb))
    println("  $(length(nb.cells)) cells, $(length(bad)) errored, $(round(t, digits=1)) s → html/$(basename(out))")
    Pluto.SessionActions.shutdown(session, nb; async=false)
end
exit(nbad == 0 ? 0 : 1)
