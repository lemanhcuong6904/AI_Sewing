import numpy as np
import cv2


def deccode_output_score_and_ptss(tpMap, topk_n=200, ksize=3):
    b, c, h, w = tpMap.shape
    assert b == 1, 'only support bsize==1'
    displacement = tpMap[:, 1:5, :, :][0]
    center = tpMap[:, 0, :, :][0]
    
    # sigmoid
    heat = 1.0 / (1.0 + np.exp(-center))
    
    # max pool
    kernel = np.ones((ksize, ksize), dtype=np.float32)
    hmax = cv2.dilate(heat, kernel)
    keep = (hmax == heat).astype(np.float32)
    heat = heat * keep
    heat_flat = heat.flatten()

    topk_n = min(topk_n, heat_flat.size)
    if topk_n > 0:
        indices = np.argpartition(heat_flat, -topk_n)[-topk_n:]
        indices = indices[np.argsort(heat_flat[indices])[::-1]]
        scores = heat_flat[indices]

        yy = indices // w
        xx = indices % w
        ptss = np.stack((yy, xx), axis=-1)
    else:
        ptss = np.empty((0, 2), dtype=np.int32)
        scores = np.empty((0,), dtype=np.float32)

    displacement = displacement.transpose((1, 2, 0))
    return ptss, scores, displacement


def postprocess_mlsd_squares(output_tensors, img_shape, input_shape=(512, 512),
                             params={'score': 0.04,             # Default 0.06. Lowered to accept more fragment candidates
                                     'outside_ratio': 0.50,     # Default 0.28. Increased to tolerate gaps before intersection
                                     'inside_ratio': 0.60,      # Default 0.45. Increased to tolerate longer internal overlaps
                                     'w_overlap': 0.0,
                                     'w_degree': 1.95,
                                     'w_length': 0.0,
                                     'w_area': 1.86,
                                     'w_center': 0.14}):
    h_original, w_original = img_shape
    outputs = output_tensors[0]

    pts, pts_score, vmap = deccode_output_score_and_ptss(outputs, 200, 3)
    start = vmap[:, :, :2]  # (x, y)
    end = vmap[:, :, 2:]  # (x, y)
    dist_map = np.sqrt(np.sum((start - end) ** 2, axis=-1))

    junc_list = []
    segments_list = []
    for junc, score in zip(pts, pts_score):
        y, x = junc
        distance = dist_map[y, x]
        if score > params['score'] and distance > 20.0:
            junc_list.append([x, y])
            disp_x_start, disp_y_start, disp_x_end, disp_y_end = vmap[y, x, :]
            d_arrow = 1.0
            x_start = x + d_arrow * disp_x_start
            y_start = y + d_arrow * disp_y_start
            x_end = x + d_arrow * disp_x_end
            y_end = y + d_arrow * disp_y_end
            segments_list.append([x_start, y_start, x_end, y_end])

    if len(segments_list) == 0:
        return np.empty((0, 4), dtype=np.float32)

    segments = np.array(segments_list)

    ####### post processing for squares
    # 1. get unique lines
    point = np.array([[0, 0]])
    point = point[0]
    start = segments[:, :2]
    end = segments[:, 2:]
    diff = start - end
    a = diff[:, 1]
    b = -diff[:, 0]
    c = a * start[:, 0] + b * start[:, 1]

    d = np.abs(a * point[0] + b * point[1] - c) / np.sqrt(a ** 2 + b ** 2 + 1e-10)
    theta = np.arctan2(diff[:, 0], diff[:, 1]) * 180 / np.pi
    theta[theta < 0.0] += 180
    hough = np.concatenate([d[:, None], theta[:, None]], axis=-1)

    d_quant = 4       # Default 1. Increased to merge lines up to 2 pixels apart
    theta_quant = 3   # Default 2. Increased to merge lines up to 3 degrees apart
    hough[:, 0] //= d_quant
    hough[:, 1] //= theta_quant
    _, indices, counts = np.unique(hough, axis=0, return_index=True, return_counts=True)

    acc_map = np.zeros([512 // d_quant + 1, 360 // theta_quant + 1], dtype='float32')
    idx_map = np.zeros([512 // d_quant + 1, 360 // theta_quant + 1], dtype='int32') - 1
    yx_indices = hough[indices, :].astype('int32')
    acc_map[yx_indices[:, 0], yx_indices[:, 1]] = counts
    idx_map[yx_indices[:, 0], yx_indices[:, 1]] = indices

    acc_map_np = acc_map

    ### fast suppression using numpy/opencv
    kernel = np.ones((5, 5), dtype=np.float32)
    max_acc_map = cv2.dilate(acc_map_np, kernel)
    acc_map_np = acc_map_np * (acc_map_np == max_acc_map).astype(np.float32)
    flatten_acc_map = acc_map_np.flatten()

    topk_n = min(len(pts), flatten_acc_map.size)
    if topk_n > 0:
        indices_flat = np.argpartition(flatten_acc_map, -topk_n)[-topk_n:]
        indices_flat = indices_flat[np.argsort(flatten_acc_map[indices_flat])[::-1]]
        topk_values = flatten_acc_map[indices_flat]
        
        _, w_acc = acc_map_np.shape
        yy = indices_flat // w_acc
        xx = indices_flat % w_acc
        yx = np.stack((yy, xx), axis=-1)
    else:
        yx = np.empty((0, 2), dtype=np.int32)
        topk_values = np.empty((0,), dtype=np.float32)
    indices = idx_map[yx[:, 0], yx[:, 1]]
    basis = 5 // 2

    merged_segments = []
    for yx_pt, max_indice, value in zip(yx, indices, topk_values):
        y, x = yx_pt
        if max_indice == -1 or value == 0:
            continue
        segment_list = []
        for y_offset in range(-basis, basis + 1):
            for x_offset in range(-basis, basis + 1):
                indice = idx_map[y + y_offset, x + x_offset]
                cnt = int(acc_map_np[y + y_offset, x + x_offset])
                if indice != -1:
                    segment_list.append(segments[indice])
                if cnt > 1:
                    check_cnt = 1
                    current_hough = hough[indice]
                    for new_indice, new_hough in enumerate(hough):
                        if (current_hough == new_hough).all() and indice != new_indice:
                            segment_list.append(segments[new_indice])
                            check_cnt += 1
                        if check_cnt == cnt:
                            break
        group_segments = np.array(segment_list).reshape([-1, 2])
        sorted_group_segments = np.sort(group_segments, axis=0)
        x_min, y_min = sorted_group_segments[0, :]
        x_max, y_max = sorted_group_segments[-1, :]

        deg = theta[max_indice]
        if deg >= 90:
            merged_segments.append([x_min, y_max, x_max, y_min])
        else:
            merged_segments.append([x_min, y_min, x_max, y_max])

    if len(merged_segments) == 0:
        return np.empty((0, 4), dtype=np.float32)

    # 2. get intersections
    new_segments = np.array(merged_segments)  # (x1, y1, x2, y2)
    start = new_segments[:, :2]  # (x1, y1)
    end = new_segments[:, 2:]  # (x2, y2)
    new_centers = (start + end) / 2.0
    diff = start - end
    dist_segments = np.sqrt(np.sum(diff ** 2, axis=-1))

    # ax + by = c
    a = diff[:, 1]
    b = -diff[:, 0]
    c = a * start[:, 0] + b * start[:, 1]
    pre_det = a[:, None] * b[None, :]
    det = pre_det - np.transpose(pre_det)

    pre_inter_y = a[:, None] * c[None, :]
    inter_y = (pre_inter_y - np.transpose(pre_inter_y)) / (det + 1e-10)
    pre_inter_x = c[:, None] * b[None, :]
    inter_x = (pre_inter_x - np.transpose(pre_inter_x)) / (det + 1e-10)
    inter_pts = np.concatenate([inter_x[:, :, None], inter_y[:, :, None]], axis=-1).astype('int32')

    # 3. get corner information
    dist_inter_to_segment1_start = np.sqrt(np.sum(((inter_pts - start[:, None, :]) ** 2), axis=-1, keepdims=True))
    dist_inter_to_segment1_end = np.sqrt(np.sum(((inter_pts - end[:, None, :]) ** 2), axis=-1, keepdims=True))
    dist_inter_to_segment2_start = np.sqrt(np.sum(((inter_pts - start[None, :, :]) ** 2), axis=-1, keepdims=True))
    dist_inter_to_segment2_end = np.sqrt(np.sum(((inter_pts - end[None, :, :]) ** 2), axis=-1, keepdims=True))

    dist_inter_to_segment1 = np.sort(np.concatenate([dist_inter_to_segment1_start, dist_inter_to_segment1_end], axis=-1), axis=-1)
    dist_inter_to_segment2 = np.sort(np.concatenate([dist_inter_to_segment2_start, dist_inter_to_segment2_end], axis=-1), axis=-1)

    inter_to_start = new_centers[:, None, :] - inter_pts
    deg_inter_to_start = np.arctan2(inter_to_start[:, :, 1], inter_to_start[:, :, 0]) * 180 / np.pi
    deg_inter_to_start[deg_inter_to_start < 0.0] += 360
    inter_to_end = new_centers[None, :, :] - inter_pts
    deg_inter_to_end = np.arctan2(inter_to_end[:, :, 1], inter_to_end[:, :, 0]) * 180 / np.pi
    deg_inter_to_end[deg_inter_to_end < 0.0] += 360

    deg1_map, deg2_map = deg_inter_to_start, deg_inter_to_end
    deg_sort = np.sort(np.concatenate([deg1_map[:, :, None], deg2_map[:, :, None]], axis=-1), axis=-1)

    deg_diff_map = np.abs(deg1_map - deg2_map)
    deg_diff_map[deg_diff_map > 180] = 360 - deg_diff_map[deg_diff_map > 180]

    deg_range = [60, 120]
    corner_dict = {corner_info: [] for corner_info in range(4)}
    
    for i in range(inter_pts.shape[0]):
        for j in range(i + 1, inter_pts.shape[1]):
            x, y = inter_pts[i, j, :]
            deg1, deg2 = deg_sort[i, j, :]
            deg_diff = deg_diff_map[i, j]

            check_degree = deg_diff > deg_range[0] and deg_diff < deg_range[1]

            outside_ratio = params['outside_ratio']
            inside_ratio = params['inside_ratio']
            check_distance = ((dist_inter_to_segment1[i, j, 1] >= dist_segments[i] and \
                               dist_inter_to_segment1[i, j, 0] <= dist_segments[i] * outside_ratio) or \
                              (dist_inter_to_segment1[i, j, 1] <= dist_segments[i] and \
                               dist_inter_to_segment1[i, j, 0] <= dist_segments[i] * inside_ratio)) and \
                             ((dist_inter_to_segment2[i, j, 1] >= dist_segments[j] and \
                               dist_inter_to_segment2[i, j, 0] <= dist_segments[j] * outside_ratio) or \
                              (dist_inter_to_segment2[i, j, 1] <= dist_segments[j] and \
                               dist_inter_to_segment2[i, j, 0] <= dist_segments[j] * inside_ratio))

            if check_degree and check_distance:
                if (deg1 >= 0 and deg1 <= 45 and deg2 >= 45 and deg2 <= 120) or \
                        (deg2 >= 315 and deg1 >= 45 and deg1 <= 120):
                    corner_info = 0
                elif (deg1 >= 45 and deg1 <= 125 and deg2 >= 125 and deg2 <= 225):
                    corner_info = 1
                elif (deg1 >= 125 and deg1 <= 225 and deg2 >= 225 and deg2 <= 315):
                    corner_info = 2
                elif (deg1 >= 0 and deg1 <= 45 and deg2 >= 225 and deg2 <= 315) or \
                        (deg2 >= 315 and deg1 >= 225 and deg1 <= 315):
                    corner_info = 3
                else:
                    continue
                corner_dict[corner_info].append([x, y, i, j])

    square_list = []
    connect_list = []
    segments_list = []
    for corner0 in corner_dict[0]:
        for corner1 in corner_dict[1]:
            connect01 = False
            for corner0_line in corner0[2:]:
                if corner0_line in corner1[2:]:
                    connect01 = True; break
            if connect01:
                for corner2 in corner_dict[2]:
                    connect12 = False
                    for corner1_line in corner1[2:]:
                        if corner1_line in corner2[2:]:
                            connect12 = True; break
                    if connect12:
                        for corner3 in corner_dict[3]:
                            connect23 = False
                            for corner2_line in corner2[2:]:
                                if corner2_line in corner3[2:]:
                                    connect23 = True; break
                            if connect23:
                                for corner3_line in corner3[2:]:
                                    if corner3_line in corner0[2:]:
                                        square_list.append(corner0[:2] + corner1[:2] + corner2[:2] + corner3[:2])
                                        connect_list.append([corner0_line, corner1_line, corner2_line, corner3_line])
                                        segments_list.append(corner0[2:] + corner1[2:] + corner2[2:] + corner3[2:])

    squares = []
    score_array = []
    
    if len(square_list) > 0:
        try:
            map_size = input_shape[0] / 2
            squares = np.array(square_list).reshape([-1, 4, 2])
            connect_array = np.array(connect_list)
            segments_array = np.array(segments_list).reshape([-1, 4, 2])

            squares_rollup = np.roll(squares, 1, axis=1)
            squares_rolldown = np.roll(squares, -1, axis=1)
            vec1 = squares_rollup - squares
            normalized_vec1 = vec1 / (np.linalg.norm(vec1, axis=-1, keepdims=True) + 1e-10)
            vec2 = squares_rolldown - squares
            normalized_vec2 = vec2 / (np.linalg.norm(vec2, axis=-1, keepdims=True) + 1e-10)
            inner_products = np.sum(normalized_vec1 * normalized_vec2, axis=-1)
            squares_degree = np.arccos(inner_products) * 180 / np.pi

            overlap_scores = []
            degree_scores = []
            length_scores = []

            def check_outside_inside(segments_info, connect_idx):
                if connect_idx == segments_info[0]:
                    check_dist_mat = dist_inter_to_segment1
                else:
                    check_dist_mat = dist_inter_to_segment2
                i, j = segments_info
                min_dist, max_dist = check_dist_mat[i, j, :]
                connect_dist = dist_segments[connect_idx]
                if max_dist > connect_dist: return 'outside', min_dist, 0, 1
                else: return 'inside', min_dist, -1, -1

            for connects, segments_sq, square, degree in zip(connect_array, segments_array, squares, squares_degree):
                cover = 0
                perimeter = 0
                square_length = []
                for start_idx in range(4):
                    end_idx = (start_idx + 1) % 4
                    connect_idx = connects[start_idx]
                    start_segments = segments_sq[start_idx]
                    end_segments = segments_sq[end_idx]
                    _, start_min, start_cover_param, start_peri_param = check_outside_inside(start_segments, connect_idx)
                    _, end_min, end_cover_param, end_peri_param = check_outside_inside(end_segments, connect_idx)
                    cover += dist_segments[connect_idx] + start_cover_param * start_min + end_cover_param * end_min
                    perimeter += dist_segments[connect_idx] + start_peri_param * start_min + end_peri_param * end_min
                    square_length.append(dist_segments[connect_idx] + start_peri_param * start_min + end_peri_param * end_min)
                overlap_scores.append(cover / perimeter)

                deg0, deg1, deg2, deg3 = degree
                deg_ratio1 = deg0 / deg2 if deg2 > deg0 else deg2 / deg0
                deg_ratio2 = deg1 / deg3 if deg3 > deg1 else deg3 / deg1
                degree_scores.append((deg_ratio1 + deg_ratio2) / 2)

                len0, len1, len2, len3 = square_length
                len_ratio1 = len0 / len2 if len2 > len0 else len2 / len0
                len_ratio2 = len1 / len3 if len3 > len1 else len3 / len1
                length_scores.append((len_ratio1 + len_ratio2) / 2)

            overlap_scores = np.array(overlap_scores)
            overlap_scores /= np.max(overlap_scores)
            degree_scores = np.array(degree_scores)
            length_scores = np.array(length_scores)

            area_scores = np.reshape(squares, [-1, 4, 2])
            area_x = area_scores[:, :, 0]
            area_y = area_scores[:, :, 1]
            correction = area_x[:, -1] * area_y[:, 0] - area_y[:, -1] * area_x[:, 0]
            area_scores = np.sum(area_x[:, :-1] * area_y[:, 1:], axis=-1) - np.sum(area_y[:, :-1] * area_x[:, 1:], axis=-1)
            area_scores = 0.5 * np.abs(area_scores + correction)
            area_scores /= (map_size * map_size)

            centers = np.array([[256 // 2, 256 // 2]], dtype='float32')
            square_centers = np.mean(squares, axis=1)
            center2center = np.sqrt(np.sum((centers - square_centers) ** 2))
            center_scores = center2center / (map_size / np.sqrt(2.0))

            score_array = params['w_overlap'] * overlap_scores \
                          + params['w_degree'] * degree_scores \
                          + params['w_area'] * area_scores \
                          - params['w_center'] * center_scores \
                          + params['w_length'] * length_scores

            sorted_idx = np.argsort(score_array)[::-1]
            score_array = score_array[sorted_idx]
            squares = squares[sorted_idx]
        except Exception:
            squares = []
            score_array = []

    # Final scaling logic
    h_ratio, w_ratio = h_original / input_shape[0], w_original / input_shape[1]
    
    # If a square is found, we convert the top square into 4 line segments.
    # Otherwise, fallback to the merged cluster segments (the edges found so far)
    if len(squares) > 0:
        best_square = squares[0] # (4, 2) array of the best square's corners
        lines_to_return = []
        for i in range(4):
            x1, y1 = best_square[i]
            x2, y2 = best_square[(i+1)%4]
            lines_to_return.append([x1, y1, x2, y2])
        lines = 2 * np.array(lines_to_return, dtype=np.float32)
    else:
        lines = 2 * np.array(new_segments, dtype=np.float32)

    # Rescale to original image dimension
    if len(lines) > 0:
        lines[:, 0] = lines[:, 0] * w_ratio
        lines[:, 1] = lines[:, 1] * h_ratio
        lines[:, 2] = lines[:, 2] * w_ratio
        lines[:, 3] = lines[:, 3] * h_ratio
        
    return lines


from concurrent.futures import ThreadPoolExecutor

def postprocess_mlsd_squares_batch(output_tensors, img_shapes, input_shape=(512, 512)):
    """
    Batched postprocessing for M-LSD using ThreadPoolExecutor.
    output_tensors: list of tensors where each tensor has shape (B, ...)
    img_shapes: list of (H, W) for each batch element
    """
    B = output_tensors[0].shape[0]
    
    def process_single(i):
        # Extract the i-th element and keep the batch dimension of 1
        single_outputs = [tensor[i:i+1] for tensor in output_tensors]
        return postprocess_mlsd_squares(single_outputs, img_shapes[i], input_shape=input_shape)
        
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_single, range(B)))
        
    return results

