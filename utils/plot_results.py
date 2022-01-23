def plot_loss(title='loss',dst_directory = './comparison_results/', curves):
    Path(dst_directory).mkdir(parents=True, exist_ok=True)
    for lable, array in curves:
        plt.plot(np.arange(1, epochs+1), np.array(array).flatten(), label='{}'.format(lable))

    plt.title(title)
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('#epochs')
    plt.yscale('log')
    plt.savefig(dst_directory + title +'.png')
    plt.show()
