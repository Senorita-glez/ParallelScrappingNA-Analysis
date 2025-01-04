def displayR(prices, emotions1, emotions2):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(10, 8))
    grid = GridSpec(4, 6, figure=fig)

    fig.suptitle("Análisis de Precios y Emociones", fontsize=20)

    ax1 = fig.add_subplot(grid[0:4, 0:4])
    ax1.set_title("Histograma de Precios")
    ax1.hist(prices, bins=10, color="blue", edgecolor="black")
    ax1.set_xlabel("Precios")
    ax1.set_ylabel("Frecuencia")

    ax2 = fig.add_subplot(grid[0:2, 4:6])
    ax2.set_title("Porcentaje de Emociones (Dataset 1)")
    ax2.bar(
        ["Positivo", "Negativo"],
        [emotions1["positive"], emotions1["negative"]],
        color=["orange", "red"],
    )
    ax2.set_ylabel("Porcentaje (%)")

    ax3 = fig.add_subplot(grid[2:4, 4:6])
    ax3.set_title("Porcentaje de Emociones (Dataset 2)")
    ax3.bar(
        ["Positivo", "Negativo"],
        [emotions2["positive"], emotions2["negative"]],
        color=["green", "purple"],
    )
    ax3.set_ylabel("Porcentaje (%)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
