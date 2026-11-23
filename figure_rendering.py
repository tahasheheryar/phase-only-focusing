"""Plotting functions for the archived WolfX numerical results."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from scipy.optimize import brentq, minimize_scalar

DARK_COUNTS = 5.0


def receiver_gain(records, key):
    initial = np.asarray([record["projected_takagi"] for record in records])
    values = np.asarray([record[key] for record in records])
    return 100 * (values / initial - 1)


def count_receiver_wins(records, key):
    return sum(record["one_step_update"] > record[key] + 1e-9 for record in records)


BLUE, ORANGE, GREEN, PURPLE, GRAY = "#24688e", "#ba562c", "#477d57", "#745b92", "#59636c"


LAWS = {"gaussian": ("Gaussian", BLUE), "clipped_1p5": ("Clipped Gaussian", ORANGE),
        "unit_magnitude": ("Equal magnitudes", GREEN)}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 8.5,
        "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .7,
        "legend.frameon": False, "legend.fontsize": 7.3,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
        "axes.labelpad": 4, "xtick.direction": "out", "ytick.direction": "out",
    })


def panels(rows=2, height=5.45):
    fig, ax = plt.subplots(rows, 2, figsize=(7.25, height), layout="constrained")
    flat = ax.ravel()
    for label, panel in zip("abcdef", flat):
        panel.set_title(label, loc="left", fontweight="bold")
    return fig, flat


def diagram(ax):
    ax.set(xlim=(0,1), ylim=(0,1))
    ax.axis("off")


def box(ax, x, y, w, h, label, color=BLUE):
    rect = FancyBboxPatch((x,y),w,h, boxstyle="round,pad=0.015",
                           edgecolor=color, facecolor="white", linewidth=.85)
    ax.add_patch(rect)
    ax.text(x+w/2,y+h/2,label,color=color,ha="center",va="center")


def arrow(ax, point_a, point_b, color=GRAY, alpha=1):
    ax.annotate("", xy=point_b, xytext=point_a,
                arrowprops=dict(arrowstyle="->",color=color,lw=.8,alpha=alpha,
                                shrinkA=1,shrinkB=1))


def sem(x):
    x = np.asarray(x)
    return x.std(axis=0, ddof=1)/np.sqrt(len(x))


def ratio_ci(x, y):
    x, y = np.array(x), np.array(y)
    draws = np.random.default_rng(10733).integers(len(x),size=(10000,len(x)))
    ratio = x[draws].mean(axis=1)/y[draws].mean(axis=1)
    return np.array([x.mean()/y.mean(), *np.quantile(ratio,[.025,.975])])


def interval_points(ax, x, records, color=BLUE, marker="o"):
    mean, lo, hi = np.asarray(records).T
    ax.errorbar(x,mean,yerr=[np.maximum(0,mean-lo),np.maximum(0,hi-mean)],
                linestyle="none",marker=marker,color=color,capsize=2.5,markersize=4)


def paired_plot(ax, values, labels, ylabel, colors):
    values = np.array(values)
    x = np.arange(values.shape[1])
    for row in values:
        ax.plot(x,row,color=GRAY,alpha=.19,lw=.65)
    for i, color in enumerate(colors):
        ax.scatter(np.full(len(values),i),values[:,i],s=9,color=color,alpha=.38,zorder=3)
        ax.errorbar(i,values[:,i].mean(),yerr=sem(values[:,i]),fmt="D",color=color,
                    markeredgecolor="white",ms=5,capsize=2,zorder=4)
    ax.set(xticks=x,xticklabels=labels,ylabel=ylabel)
    ax.margins(x=.1)


def by_drive(record, drive):
    return next(row for row in record["results"] if np.isclose(row["drive"],drive))


def figure1(data):
    fig, axs = panels(rows=3, height=7.85)
    ax = axs[0]
    diagram(ax)
    ax.text(.5,.97,"Phase-only control of second-harmonic light",ha="center",va="top")
    box(ax,.04,.36,.18,.24,"Input\nphases")
    for y in (.18,.32,.48,.64,.78):
        ax.add_patch(Circle((.49,y),.024,facecolor=GREEN,edgecolor="white",lw=.75,zorder=3))
        arrow(ax,(.22,.48),(.466,y),BLUE,.73)
        arrow(ax,(.514,y),(.78,.48),ORANGE,.73)
    box(ax,.78,.36,.18,.24,"One\nreceiver",ORANGE)
    for x, label, color in ((.23,"Pump",BLUE),(.47,"Nonlinear sites",GREEN),(.80,"Second harmonic",ORANGE)):
        ax.text(x,.06,label,ha="center",color=color,fontsize=7.3)

    ax = axs[1]
    diagram(ax)
    ax.text(.5,.91,"Shared-scatterer model",ha="center",weight="medium")
    ax.text(.5,.74,r"$u_k=(Ax)_k,\qquad |x_j|=N^{-1/2}$",ha="center",fontsize=10)
    ax.text(.5,.57,r"$q=(KN)^{-1/2}\sum_k z_k u_k^2=x^{\mathsf{T}}Tx$",ha="center",fontsize=10)
    ax.text(.5,.44,r"$T=A^{\mathsf{T}}\,\mathrm{diag}(z)\,A/\sqrt{KN}$",ha="center")
    box(ax,.05,.14,.37,.17,"Single bright site",GREEN)
    box(ax,.58,.14,.37,.17,"Collective response",ORANGE)

    ax = axs[2]
    n = 1024
    rows = [r for r in data["dense_fresh"] if r["n"]==n]
    values = [[r[k]/(n*r["background"]) for k in ("projected","corrected","best")] for r in rows]
    paired_plot(ax,values,["Projected\nTakagi","Analytic\nupdate","Numerical\nmask"],
                r"Enhancement per input, $\eta/N$",[BLUE,GREEN,ORANGE])
    ax.axhline(np.pi**2/8,color=GRAY,ls="--",lw=.8,label=r"Projected Takagi, $\pi^2/8$")
    ax.set_title(r"Independent Gaussian entries, $N=1{,}024$")
    ax.legend(loc="upper left")

    ax = axs[3]
    r = data["dense_reference"]
    sizes = [a["modes"] for a in r["rows"]]
    lo = r["analytic_lower"]["asymptotic_enhancement_per_mode"]
    hi = r["analytic_upper"]["continuous_enhancement_per_mode"]
    ax.axhspan(lo,hi,color=GREEN,alpha=.12,label="Asymptotic bounds on optimum")
    ax.errorbar(sizes,[a["summary"]["feasible"]["mean"] for a in r["rows"]],
                yerr=[a["summary"]["feasible"]["standard_error"] for a in r["rows"]],
                fmt="o-",ms=4,lw=1,capsize=2,color=BLUE,label="Feasible finite-size mean")
    ax.axhline(np.pi**2/8,color=GRAY,ls="--",lw=.8)
    ax.set(xscale="log",xlabel="Input modes, $N$",ylabel=r"Enhancement per input, $\eta/N$")
    ax.set_title("Independent Gaussian entries")
    ax.legend(loc="lower right")
    optical_gain_panels(axs[4], axs[5], data["optical"])
    return fig


def optical_gain_panels(cumulative, paired, records):
    curves = (
        ("one_step_update", "Analytic update", BLUE, "-"),
        ("field_gradient_matched_rms", "Matched field step", GREEN, "--"),
        ("intensity_gradient_fixed_rms", "Fixed intensity step", PURPLE, ":"),
    )
    gains = {}
    for key, label, color, linestyle in curves:
        values = receiver_gain(records, key)
        gains[key] = values
        sorted_gains = np.sort(values)
        cumulative.step(
            np.r_[sorted_gains[0], sorted_gains, 40],
            np.r_[0, np.arange(1, len(values) + 1) / len(values), 1],
            where="post", lw=1.1, color=color, linestyle=linestyle, label=label,
        )
    cumulative.axvline(0, color=GRAY, lw=.65)
    cumulative.set(xlim=(-1.5, 40), ylim=(0, 1.02),
                   xticks=[0, 10, 20, 30, 40],
                   xlabel="Predicted gain over projected Takagi (%)",
                   ylabel="Fraction of receivers",
                   title="Optical receiver tensors")
    cumulative.legend(loc="lower right")

    field = gains["field_gradient_matched_rms"]
    analytic = gains["one_step_update"]
    paired.scatter(field, analytic, s=7, color=BLUE, alpha=.42, edgecolors="none")
    paired.plot([-1.5, 40], [-1.5, 40], "--", color=GRAY, lw=.75)
    paired.set(xlim=(-1.5, 40), ylim=(-1.5, 40), xticks=[0, 10, 20, 30, 40],
               yticks=[0, 10, 20, 30, 40], xlabel="Gradient step gain (%)",
               ylabel="Analytic update gain (%)", title="Paired comparison at equal step size")
    wins = count_receiver_wins(records, "field_gradient_matched_rms")
    paired.text(.96, .065, f"Analytic update higher at {wins} of {len(records)}",
                transform=paired.transAxes, ha="right", va="bottom", fontsize=7)


def figure2(data):
    fig, axs = panels()
    ax = axs[0]
    diagram(ax)
    ax.text(.5,.93,"Gaussian collection weights",ha="center")
    ax.add_patch(Rectangle((.35,.26),.30,.49,color=GRAY,alpha=.10))
    arrow(ax,(.05,.48),(.95,.48))
    ax.text(.18,.64,"Brightest-site\nregime",ha="center",color=BLUE)
    ax.text(.50,.64,"Crossover",ha="center",color=GRAY)
    ax.text(.81,.64,"Collective\nresponse",ha="center",color=ORANGE)
    for x,label in ((.18,r"$\ll 1$"),(.50,r"$\asymp 1$"),(.81,r"$\gg 1$")):
        ax.text(x,.35,label,ha="center")
    ax.text(.50,.13,r"$K/(N\log K)$",ha="center",fontsize=10)

    ax=axs[1]
    rows = [r for r in data["sparse"] if 2*r["k"]==r["n"]]
    n = [r["n"] for r in rows]
    ax.plot(n,[r["leading_asymptote"] for r in rows],ls="--",color=GRAY,label="Leading asymptote")
    ax.plot(n,[r["exact_ensemble_mean"] for r in rows],color=GREEN,label="Exact finite mean")
    ax.scatter(n,[r["simulated_mean"] for r in rows],s=24,color=BLUE,facecolor="white",zorder=4,
               label="Simulated mean")
    ax.set(xscale="log",xlabel="Input modes, $N$",ylabel=r"Enhancement per input, $\eta/N$",title="Brightest-weight site at $K=N/2$")
    ax.legend(loc="lower right")

    ax=axs[2]
    for r in [x for x in data["crossover"] if x["n"]==192]:
        label,color=LAWS[r["distribution"]]
        ax.plot(r["ratios"],r["mean"],".-",color=color,label=label)
        ax.fill_between(r["ratios"],r["ci_low"],r["ci_high"],color=color,alpha=.13)
        ax.errorbar(r["half"],.5,xerr=[[r["half"]-r["bootstrap_half_ci"][0]],
                                     [r["bootstrap_half_ci"][1]-r["half"]]],fmt="o",
                    color=color,markerfacecolor="white",ms=4,capsize=2)
    ax.axhline(.5,color=GRAY,ls="--",lw=.7)
    ax.set(xscale="log",ylim=(0,1.05),xlabel=r"Scatterers per input, $K/N$",
           ylabel="One-site / best found intensity",title="$N=192$")
    ax.legend()

    ax=axs[3]
    rows={r["distribution"]:r for r in data["passive_collection"]}
    pair=np.column_stack([[a["single_over_best"] for a in rows[name]["raw"]]
                          for name in ("speckled","flat")])
    paired_plot(ax,pair,["Uneven","Equal"],"One-site / best found intensity",[BLUE,GREEN])
    ax.set(title="$N=192$, $K=384$, $D=576$",ylim=(0,1.01),xlabel="Collection weights")
    return fig


def figure3(data):
    fig,axs=panels()
    ax=axs[0]
    p=np.linspace(0,1,180)
    for g,color in ((.15,BLUE),(2.,GREEN),(8.,ORANGE)):
        ax.plot(p,np.sqrt(p)*np.tanh(g*np.sqrt(p))/g,color=color,label=f"$g={g:g}$")
    ax.set(xlabel="Local pump fraction",ylabel=r"Harmonic amplitude magnitude / $g$",
           title="Phase-matched conversion")
    ax.legend()

    ax=axs[1]
    rows=data["passive_screened"]
    sizes=rows[0]["tested_group_sizes"]
    powers=100*np.array([r["groups"] for r in rows])
    ref=100*np.mean([r["screened"] for r in rows])
    ax.axhline(ref,ls="--",lw=.85,color=GRAY,label="Best screened one-site mean")
    ax.errorbar(sizes,powers.mean(axis=0),yerr=sem(powers),fmt="o-",ms=3,capsize=2,color=BLUE,
                label="Mean by group size")
    ax.set(xscale="log",xticks=[1,4,16,64,256],xticklabels=["1","4","16","64","256"],
           xlabel="Group size",ylabel="Collected power (% of launch)",title=r"$N=K=D=8{,}192$, $g=8$")
    ax.legend(loc="upper left")

    ax=axs[2]
    rows=[r for r in data["passive_groups"] if r["n"]==8192 and r["d"]==8192]
    sizes=rows[0]["counts"]
    img=np.array([np.bincount(np.asarray([r["power"][key] for r in rows]).argmax(axis=1),
                              minlength=len(sizes)) for key in ("0.15","8.0","12.0")])
    heat=ax.imshow(img,aspect="auto",cmap="Blues",vmin=0,vmax=len(rows))
    for i,j in np.argwhere(img>0):
        ax.text(j,i,str(img[i,j]),ha="center",va="center",color="white" if img[i,j]>12 else GRAY)
    ax.set(xticks=range(len(sizes)),xticklabels=sizes,yticks=[0,1,2],yticklabels=["0.15","8","12"],
           xlabel="Best tested group size",ylabel="Drive, $g$",title="Group choices in 24 geometries")
    fig.colorbar(heat,ax=ax,shrink=.67,label="Geometries")

    ax=axs[3]
    rows=data["passive_small"]
    values=100*np.array([[r[k] for k in ("one","weak_mask_at_drive","simple_multi","optimized")] for r in rows])
    paired_plot(ax,values,["One\nsite","Weak-drive\nmask","Group\nmask","Direct\noptimizer"],
                "Collected power (% of launch)",[GRAY,BLUE,GREEN,ORANGE])
    ax.set_title("$N=64$, $g=8$")
    return fig


def guide_optimum(g, ratio):
    def slope(p):
        x=g*np.sqrt(p)
        return g if x==0 else (g/2)*(np.tanh(x)/x+1/np.cosh(x)**2)
    if slope(1)>=ratio*g:
        return 1.
    return brentq(lambda p: slope(p)-ratio*slope(1-p),0,1)


def figure4(data):
    fig,axs=panels()
    rows=data["two_guides"]["rows"]
    ax=axs[0]
    diagram(ax)
    box(ax,.27,.39,.16,.19,"Pump\ncoupler")
    box(ax,.55,.62,.15,.16,"Guide 1",GREEN)
    box(ax,.55,.21,.15,.16,"Guide 2",GREEN)
    box(ax,.80,.39,.166,.19,"Harmonic\ncombiner",ORANGE)
    for a,b,color in (((.04,.545),(.27,.545),BLUE),((.04,.425),(.27,.425),BLUE),
                      ((.43,.52),(.55,.70),BLUE),((.43,.45),(.55,.29),BLUE),((.70,.70),(.80,.52),ORANGE),
                      ((.70,.29),(.80,.46),ORANGE)):
        arrow(ax,a,b,color)
    ax.text(.14,.59,r"$1/\sqrt{2}$",ha="center",color=BLUE)
    ax.text(.14,.34,r"$e^{i\theta}/\sqrt{2}$",ha="center",color=BLUE)
    ax.text(.46,.66,r"$p$",ha="center",color=BLUE)
    ax.text(.46,.27,r"$1-p$",ha="center",color=BLUE)
    ax.text(.50,.08,"Two-guide numerical example",ha="center")

    ax=axs[1]
    r=.90
    g=np.linspace(.025,2.,180)
    critical=brentq(lambda x: (np.tanh(x)/x+1/np.cosh(x)**2)/2-r,.05,3)
    ax.plot(g,[guide_optimum(x,r) for x in g],color=BLUE)
    ax.axvline(critical,ls="--",lw=.8,color=GRAY)
    ax.annotate(f"$g={critical:.3f}$",(critical,.58),xytext=(.53,.55),
                arrowprops=dict(arrowstyle="-",color=GRAY,lw=.7),fontsize=8)
    ax.set(xlabel="Drive, $g$",ylabel="Optimal pump fraction in guide 1",
           ylim=(.46,1.04),title="Ideal guides, field ratio 0.90")

    ax=axs[2]
    for curve,color in zip(data["two_guides"]["curves"],[BLUE,ORANGE]):
        phase=np.asarray(curve["phases"])/np.pi
        reference=curve["physical_single"]
        ratio=np.asarray(curve["collected_power"])/reference
        probes=np.asarray(curve["train_phases"])/np.pi
        detector=((np.asarray(curve["first_train"])-DARK_COUNTS)*curve["scale"]/200)/reference
        i=curve["first_selected_index"]
        replay=curve["first_selected_true_power"]/reference
        ax.plot(phase,ratio,color=color,label=f'$g={curve["drive"]:.2f}$')
        other=np.arange(len(probes))!=i
        ax.scatter(probes[other],detector[other],s=12,color=color,facecolor="white",zorder=4)
        ax.plot([probes[i]]*2,[detector[i],replay],color=color,ls=":",lw=.8)
        ax.scatter(probes[i],detector[i],s=22,color=color,facecolor="white",marker="D",zorder=5)
        ax.scatter(probes[i],replay,s=26,color=color,marker="^",edgecolor="white",linewidth=.45,zorder=5)
    ax.axhline(1,ls="--",lw=.75,color=GRAY)
    ax.set(xlabel=r"Command phase / $\pi$",ylabel="Power / best single guide",title="One geometry, 12 noisy probes")
    ax.legend(loc="upper left",handlelength=1.5)

    ax=axs[3]
    for group,curve in enumerate(data["two_guides"]["curves"]):
        subset=[r for r in rows if r["cohort"]=="near_matched" and r["drive"]==curve["drive"]]
        base=[r["best_single"] for r in subset]
        for method,vals,col,marker,delta in (
            ("12 probes",[r["settings"][1]["selected_mean_power"] for r in subset],BLUE,"o",-.1),
            ("Known equal split",[r["equal_split"] for r in subset],GREEN,"^",0),
            ("Dense sweep",[r["oracle_power"] for r in subset],ORANGE,"s",.1)):
            interval_points(ax,[group+delta],[ratio_ci(vals,base)],col,marker)
            if group==0:
                ax.plot([],[],linestyle="none",marker=marker,color=col,label=method,ms=4)
    ax.axhline(1,ls="--",lw=.75,color=GRAY)
    ax.set(xlim=(-.35,1.35),xticks=[0,1],xticklabels=["Weak drive","High drive"],
           ylabel="Power / best single guide",title="128 near-matched geometries")
    ax.legend(loc="upper left")
    return fig


def figure5(data):
    fig,axs=panels()
    square=data["square_array"]
    drives=[r["drive"] for r in square[0]["results"]]
    ax=axs[0]
    def fixed(row):
        return row["groups"][row["counts"].index(16)]
    for name,extract,col in (("Site-matched",lambda r:r["single"],GRAY),
                             ("16-site group",fixed,GREEN),
                             ("Better of two weak-drive masks",lambda r:max(r["weak"],r["old_weak"]),BLUE)):
        z=100*np.array([[extract(by_drive(r,g)) for r in square] for g in drives])
        ax.errorbar(drives,z.mean(axis=1),yerr=sem(z.T),fmt="o-",ms=3,capsize=2,label=name,color=col)
    z=100*np.array([by_drive(r,8)["optimized"] for r in square])
    ax.errorbar(8,z.mean(),yerr=sem(z),fmt="D",color=ORANGE,ms=4,capsize=2,label="Direct optimization")
    ax.set(xlabel="Drive, $g$",ylabel="Collected power (% of launch)",title="Moderate hopping")
    ax.legend(loc="upper left")

    ax=axs[1]
    ablated=[by_drive(r,8) for r in square]
    baseline=[r for r in data["coupling"] if (r["hopping"],r["disorder"])==(.45,.45)]
    hopping=[r for r in data["coupling"] if (r["hopping"],r["disorder"])==(1.6,.45)]
    born=[r["born_group"][np.argmax(r["groups"])] for r in ablated]
    points=[ratio_ci([r["best_group"] for r in rows],[r["weak"] for r in rows])
            for rows in (baseline,hopping)]
    interval_points(ax,[0,1,2],[points[0],ratio_ci(born,[r["born_weak"] for r in ablated]),points[1]])
    ax.axhline(1,ls="--",lw=.75,color=GRAY)
    ax.set(xticks=[0,1,2],xticklabels=["Moderate\nhopping","Undepleted\nreplay","Higher\nhopping only"],
           ylabel="Group / weak-drive reference",title="Matched comparisons at $g=8$")

    ax=axs[2]
    rows=[by_drive(r,16) for r in data["diffraction"]]
    z=100*np.array([[r["single"],r["groups"][r["counts"].index(4)],max(r["groups"]),
                    max(r["weak"],r["old_weak"]),r["optimized"]] for r in rows])
    paired_plot(ax,z,["One\nsite","Four\nsites","Selected\ngroup","Weak-drive\nreference","Direct\nsearch"],
                "Collected power (% of launch)",[GRAY,GREEN,ORANGE,BLUE,PURPLE])
    ax.set_title("Shared-path diffraction at $g=16$")

    ax=axs[3]
    rows=data["planar"]
    chosen=[0,1,len(rows[0]["results"])-1]
    out=[]
    for i in chosen:
        power=np.array([r["results"][i]["nominal_walkoff"] for r in rows])
        out.append(ratio_ci(power[:,1],power[:,3]))
    interval_points(ax,[0,1,2],out)
    ax.axhline(1,ls="--",lw=.75,color=GRAY)
    ax.set(xticks=[0,1,2],xticklabels=["$g=1.65$\nDepletion fit","$g=3.30$\nWeak-drive fit","$g=8$\nStress test"],
           ylabel="Group / weak-drive pulse energy",title="Planar waveguide array")
    return fig


def supplement1(data):
    fig,axs=panels()
    dense=data["dense_reference"]["rows"]
    for key,col,name in (("projected",BLUE,"Takagi mask"),("analytic_update",GREEN,"Analytic update"),
                         ("feasible",ORANGE,"Numerical mask")):
        sizes = [r["modes"] for r in dense]
        means = [r["summary"][key]["mean"] for r in dense]
        axs[0].plot(sizes,means,"o-",ms=3,color=col,label=name)
    axs[0].set(xscale="log",xlabel="$N$",ylabel=r"Mean $\eta/N$")
    axs[0].legend()
    ax=axs[1]
    for r in dense:
        if "finite_instance_upper" in r["raw"][0]:
            ax.scatter([r["modes"]]*len(r["raw"]),
                       [q["finite_instance_upper"]-q["feasible"] for q in r["raw"]],s=9,alpha=.35,color=BLUE)
    ax.set(xscale="log",xlabel="$N$",ylabel=r"Upper bound minus achieved $\eta/N$")
    ax=axs[2]
    rows=data["dense_fresh"]
    x=np.array([r["n"] for r in rows])
    y=np.array([r["background"]*r["n"]/2 for r in rows])
    ax.scatter(x,y,alpha=.5,s=10,color=BLUE,label="Fixed-tensor mean / ensemble mean")
    ax.axhline(1,color=GRAY,ls="--",lw=.75)
    ax.set(xscale="log",xlabel="$N$",ylabel="Background ratio")
    ticks=np.unique(x)
    ax.set_xticks(ticks,labels=[f"{n:,}" for n in ticks])
    ax.set_xticks([],minor=True)
    ax.legend()
    ax=axs[3]
    ax.scatter([r["background"] for r in rows],[r["background_mc"] for r in rows],s=12,color=BLUE,alpha=.55)
    x=np.array([r["background"] for r in rows]); ax.plot([x.min(),x.max()],[x.min(),x.max()],"--",lw=.75,color=GRAY)
    ax.set(xlabel="Analytic phase-random mean",ylabel="Monte Carlo phase-random mean",
           xscale="log",yscale="log")
    return fig


def supplement2(data):
    fig,axs=panels()
    for ax,n in zip(axs,(48,96,192,384)):
        for r in [r for r in data["crossover"] if r["n"]==n]:
            label,c=LAWS[r["distribution"]]
            ax.plot(r["ratios"],r["mean"],".-",color=c,label=label)
            ax.fill_between(r["ratios"],r["ci_low"],r["ci_high"],color=c,alpha=.13)
        ax.axhline(.5,ls="--",color=GRAY,lw=.7)
        ax.set(xscale="log",xlim=(.05,80),ylim=(0,1.06),title=f"$N={n}$",xlabel="$K/N$",
               ylabel="One-site / best found intensity")
    axs[0].legend()
    return fig


def supplement3(data):
    fig,axs=panels()
    for ax,(n,extra) in zip(axs,((2048,1.),(2048,1.5),(8192,1.),(8192,1.5))):
        rows=[r for r in data["passive_groups"] if r["n"]==n and r["d"]==int(n*extra)]
        for key,c in (("0.15",BLUE),("2.0",GREEN),("5.0",PURPLE),("8.0",ORANGE),("12.0",GRAY)):
            z=100*np.array([r["power"][key] for r in rows])
            ax.plot(rows[0]["counts"],z.mean(axis=0),"o-",ms=2.5,color=c,label=f"$g={float(key):g}$")
        ax.set(xscale="log",yscale="log",ylim=(.00018,11),xlabel="Group size",ylabel="Collected power (% of launch)",
               title=f"$N=K={n:,}$  /  $D/N={extra:g}$")
    axs[0].legend(ncol=2)
    return fig


def supplement4(data):
    fig,axs=panels()
    ax=axs[0]
    keys=((.45,.45),(1.6,.45),(.45,1.6),(1.6,1.6))
    pts=[]
    for h,d in keys:
        rows=[r for r in data["coupling"] if (r["hopping"],r["disorder"])==(h,d)]
        pts.append(ratio_ci([r["best_group"] for r in rows],[r["weak"] for r in rows]))
    interval_points(ax,range(4),pts)
    ax.axhline(1,lw=.75,ls="--",color=GRAY)
    ax.set(xticks=range(4),xticklabels=["Baseline","Higher\nhopping","Higher\ndisorder","Both higher"],
           ylabel="Group / weak-drive reference",title="Hopping and disorder varied separately")
    model=data["diffraction"]
    ax=axs[1]
    pts=[]
    for g in (.2,8,16):
        rows=[by_drive(r,g) for r in model]
        multisite=[max(p for n,p in zip(r["counts"],r["groups"]) if n>1) for r in rows]
        pts.append(ratio_ci(multisite,[r["single"] for r in rows]))
    interval_points(ax,[.2,8,16],pts)
    ax.axhline(1,lw=.75,ls="--",color=GRAY)
    ax.set(xlabel="Drive, $g$",xticks=[.2,8,16],ylabel="Multisite / one-site power",title="Shared-path diffraction")
    ax=axs[2]
    for r in data["planar"]:
        for s in r["results"]:
            a,b=s["indexes"][1],s["indexes"][-1]
            ax.scatter(s["quasi"][a]/s["quasi"][b],s["nominal_walkoff"][1]/s["nominal_walkoff"][-1],
                       s=12,color=BLUE,alpha=.6)
    lo,hi=ax.get_xlim(); ax.plot([lo,hi],[lo,hi],ls="--",lw=.75,color=GRAY)
    ax.set(xlabel="Group / weak-drive energy without walk-off",
           ylabel="Group / weak-drive energy with walk-off",title="Planar pulse model")
    ax=axs[3]
    errors=[]
    for r in data["planar"]:
        for s in r["results"]:
            if "integration_error" in s:
                errors.append([s["integration_error"],s["zero_walkoff_error"]])
    if errors:
        pairs=np.asarray(errors)
        ax.scatter(range(1,len(pairs)+1),pairs[:,0],color=BLUE,s=15,label="Refined time and propagation grids")
        ax.scatter(range(1,len(pairs)+1),pairs[:,1],color=ORANGE,s=15,label="Zero walk-off check")
    ax.set(yscale="log",xlabel="Checked geometry",xticks=range(1,len(errors)+1),
           ylabel="Absolute difference in pulse energy fraction",title="Planar solver checks")
    ax.legend()
    return fig


def supplement5(data):
    fig,axs=panels()
    rows=data["two_guides"]["rows"]
    for ax,(cohort,drive) in zip(axs,dict.fromkeys((r["cohort"],r["drive"]) for r in rows)):
        subset=[r for r in rows if (r["cohort"],r["drive"])==(cohort,drive)]
        base=[r["best_single"] for r in subset]
        counts=[x["counts"] for x in subset[0]["settings"]]
        pts=[ratio_ci([r["settings"][i]["selected_mean_power"] for r in subset],base) for i in range(len(counts))]
        interval_points(ax,counts,pts)
        ax.plot([],[],"o",ms=4,color=BLUE,label="12-probe selection")
        ax.axhline(1,ls="--",color=GRAY,lw=.7,label="Best single guide")
        ax.axhline(np.mean([r["equal_split"] for r in subset])/np.mean(base),ls=":",color=GREEN,lw=1,label="Equal split")
        ax.axhline(np.mean([r["oracle_power"] for r in subset])/np.mean(base),ls="-.",color=ORANGE,lw=.7,label="Dense phase sweep")
        ax.set(xscale="log",xticks=counts,xticklabels=counts,xlabel="Nominal counts per probe",
               ylabel="Power / best single guide",title=f'{cohort.replace("_"," ").capitalize()}, $g={drive:.2f}$')
    axs[0].legend()
    return fig


def save_figure(fig, directory, name):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    try:
        with (directory / f"{name}.pdf").open("wb") as pdf:
            fig.savefig(pdf, format="pdf")
        with (directory / f"{name}.png").open("wb") as png:
            fig.savefig(png, format="png", dpi=300)
        from IPython.display import Image, display

        display(Image(filename=str(directory / f"{name}.png"), width=760))
    finally:
        plt.close(fig)
