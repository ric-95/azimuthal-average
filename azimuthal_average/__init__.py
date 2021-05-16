from processing import take_azimuthal_average


def _take_azimuthal_average(csv_dir, write_intermediate=False, copy=True):
    """Take the azimuthal average of all the csv files within a given directory.

    Args:
        csv_dir (str): Path to directory containing """
    csv_list = dl.get_csv_files_list(csv_dir)
    total_slices = len(csv_list)
    slices_df_list = []
    for slice_number, csv_file in enumerate(csv_list):
        print("Transforming", csv_file)
        slice_df = dl.load_single_csv_file_as_df(csv_file)
        theta = (slice_number/total_slices)*np.pi*2
        cylindrical_slice_df = transform_df_to_cylindric_coordinates(slice_df,
                                                                     theta,
                                                                     copy)
        if write_intermediate:
            intermediate_dir = os.path.join(csv_dir, "intermediate")
            if not os.path.isdir(intermediate_dir):
                os.mkdir(intermediate_dir)
            intermediate_file = os.path.join(intermediate_dir, f"cyl_slice_{slice_number}.csv")
            cylindrical_slice_df.to_csv(intermediate_file, sep=",")
        slices_df_list.append(cylindrical_slice_df)
    concatenated_slices_df = pd.concat(
        slices_df_list,
        keys=[i for i in range(total_slices)])
    print(concatenated_slices_df)
    return concatenated_slices_df.groupby(level=1).mean()
