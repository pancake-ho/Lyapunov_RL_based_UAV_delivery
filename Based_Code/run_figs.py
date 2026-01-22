from agent import run_fig23, run_fig45, run_fig67

if __name__ == "__main__":
    # 저장 폴더는 자유롭게 변경 가능
    outdir = "./results_figs"

    # Fig. 2~3
    run_fig23(episodes=30, save_dir=outdir)

    # Fig. 4~5
    run_fig45(episodes=30, save_dir=outdir)

    # Fig. 6~7
    run_fig67(episodes=30, save_dir=outdir)

    print("Done. Check PNG files under", outdir)