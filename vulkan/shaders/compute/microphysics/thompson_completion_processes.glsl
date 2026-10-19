
vec2 cloud_diameter(vec2 mass, vec2 number)
{
    vec2 coordinate = df_div(vec2(1000000000.0, 0.0), number);
    int  shape      = coordinate.x >= 13.0 ? 15 : min(15, nonnegative_floor(df_add(coordinate, vec2(0.5, 0.0))) + 2);
    vec2 ratio = vec2(float((shape + 1) * (shape + 2) * (shape + 3)), 0), s = vec2(float(shape), 0);
    vec2 lam =
        root3(df_div(df_mul(df_mul(number,
                                   df_div(df_mul(vec2(3.1415927410125732, -8.7412573179790343e-08), vec2(1000.0, 0.0)),
                                          vec2(6.0, 0.0))),
                            ratio),
                     mass));
    return df_min(vec2(4.9999998736893758e-05, 1.2631062442183295e-12),
                  df_max(vec2(9.9999999747524271e-07, 2.5247572468697606e-15),
                         df_div(df_add(s, vec2(3.6719999313354492, 6.8664550934016688e-08)), lam)));
}
vec2 snow_efficiency(vec2 diameter, vec2 cloud_d)
{
    vec2 first = coeff(14851889), last = coeff(14851889 + 1);
    vec2 coordinate =
        df_div(df_mul(vec2(100.0, 0.0), logarithm(df_div(diameter, first))), logarithm(df_div(last, first)));
    int isnow  = coordinate.x <= 0.0 ? 0 : coordinate.x >= 99.0 ? 99 : nonnegative_floor(coordinate);
    int icloud = df_sub(cloud_d, vec2(4.9999998736893758e-05, 1.2631062442183295e-12)).x >= 0.0
                     ? 49
                     : clamp(nonnegative_floor(df_mul(cloud_d, vec2(1000000.0, 0.0))), 1, 100) - 1;
    return coeff(14851889 + 2 + isnow * 100 + icloud);
}
void riming_process()
{
    vec2 cloud = df_mul(q[1], rho), number = df_mul(current_number(0), rho), snow = df_mul(q[4], rho),
         graupel = df_mul(q[5], rho);
    if (df_sub(cloud, vec2(9.999999960041972e-13, 3.9958027796366288e-21)).x <= 0.0 ||
        df_sub(number, vec2(2.0, 0.0)).x <= 0.0 ||
        (df_sub(snow, vec2(9.999999960041972e-13, 3.9958027796366288e-21)).x <= 0.0 &&
         df_sub(graupel, vec2(9.9999999747524271e-07, 2.5247572468697606e-15)).x < 0.0))
        return;
    vec2 cd = cloud_diameter(cloud, number);
    if (df_sub(cd, vec2(9.9999999747524271e-07, 2.5247572468697606e-15)).x <= 0.0)
        return;
    vec2 tc         = df_sub(t, vec2(273.14999389648438, 6.1035156022626325e-06)),
         rho_factor = root(df_div(
             df_div(vec2(101325.0, 0.0), df_mul(vec2(287.04998779296875, 1.2207031261368684e-05), vec2(298.0, 0.0))),
             rho));
    vec2 sm = vec2(0), sn = vec2(0), sd = vec2(0), gm = vec2(0), gn = vec2(0);
    if (df_sub(snow, vec2(9.999999960041972e-13, 3.9958027796366288e-21)).x > 0.0)
    {
        vec2 s     = df_div(snow, vec2(0.068999998271465302, 1.7285347042594879e-09)),
             st    = df_min(-vec2(0.10000000149011612, -1.4901161138336505e-09), tc),
             third = snow_moment(s, st, vec2(3.0, 0.0));
        sd         = df_div(third, s);
        if (df_sub(sd, vec2(0.00030000001424923539, -1.4249235417897882e-11)).x > 0.0)
        {
            vec2 efficiency = snow_efficiency(sd, cd),
                 moment     = snow_moment(s, st, vec2(2.5499999523162842, 4.7683715642676816e-08));
            vec2 rate_factor =
                df_mul(df_mul(df_mul(df_mul(df_mul(rho_factor, vec2(3.1415927410125732, -8.7412573179790343e-08)),
                                            vec2(0.25, 0.0)),
                                     vec2(40.0, 0.0)),
                              efficiency),
                       moment);
            sm = df_mul(cloud, rate_factor);
            sn = df_mul(number, rate_factor);
        }
    }
    if (df_sub(graupel, vec2(9.9999999747524271e-07, 2.5247572468697606e-15)).x >= 0.0)
    {
        vec2 n0 = df_real_power(
            vec2(10.0, 0.0),
            df_min(
                vec2(6.0, 0.0),
                df_max(vec2(2.0, 0.0),
                       df_add(vec2(3.0, 0.0),
                              df_mul(df_div(vec2(2.0, 0.0), vec2(7.0, 0.0)),
                                     df_add(df_div(logarithm(df_max(
                                                       vec2(9.9999997171806854e-10, 2.8281931525706872e-17), graupel)),
                                                   logarithm(vec2(10.0, 0.0))),
                                            vec2(8.0, 0.0)))))));
        vec2 am  = df_div(df_mul(vec2(3.1415927410125732, -8.7412573179790343e-08), vec2(400.0, 0.0)), vec2(6.0, 0.0)),
             lam = df_real_power(df_div(df_mul(df_mul(n0, am), vec2(6.0, 0.0)), graupel), vec2(0.25, 0.0)),
             inv = df_div(vec2(1.0, 0.0), lam);
        vec2 count     = df_div(df_mul(df_mul(df_mul(graupel, lam), lam), lam), df_mul(vec2(6.0, 0.0), am)),
             intercept = df_mul(count, lam), gd = df_mul(vec2(4.0, 0.0), inv);
        vec2 gamma_v = coeff(14862121 + 2), gamma_k = coeff(14862121 + 1);
        vec2 velocity  = df_mul(df_div(df_mul(df_mul(rho_factor, vec2(442.0, 0.0)), gamma_v), vec2(6.0, 0.0)),
                                df_real_power(inv, vec2(0.88999998569488525, 1.4305114759416426e-08)));
        vec2 viscosity = df_mul(df_sub(df_add(vec2(1.718000054359436, -5.4359436063577959e-08),
                                              df_mul(vec2(0.0049000000581145287, -5.811452881421264e-11), tc)),
                                       df_mul(df_mul(vec2(1.2000000424450263e-05, -4.2445026307699404e-13), tc), tc)),
                                vec2(9.9999997473787516e-06, 2.5262124918247909e-13));
        vec2 stokes    = df_div(df_mul(df_mul(df_mul(cd, cd), velocity), vec2(1000.0, 0.0)),
                                df_mul(df_mul(vec2(9.0, 0.0), viscosity), gd));
        vec2 efficiency =
            stokes.x < .4 ? vec2(0)
            : stokes.x <= 10.0
                ? df_div(df_mul(vec2(0.55000001192092896, -1.1920928910669204e-08),
                                logarithm(df_mul(vec2(2.5099999904632568, 9.5367429508996793e-09), stokes))),
                         logarithm(vec2(10.0, 0.0)))
                : vec2(0.76999998092651367, 1.9073486345888568e-08);
        vec2 rate_factor = df_mul(
            df_mul(df_mul(df_mul(df_mul(df_mul(df_mul(rho_factor, vec2(3.1415927410125732, -8.7412573179790343e-08)),
                                               vec2(0.25, 0.0)),
                                        vec2(442.0, 0.0)),
                                 gamma_k),
                          efficiency),
                   intercept),
            df_real_power(inv, vec2(3.8900001049041748, -1.0490417468034252e-07)));
        gm = df_mul(cloud, rate_factor);
        gn = df_mul(number, rate_factor);
    }
    vec2 mass = df_add(sm, gm), count = df_add(sn, gn), scale = vec2(1, 0);
    if (mass.x > 0.0)
        scale = df_min(scale, df_div(cloud, df_mul(step, mass)));
    if (count.x > 0.0)
        scale = df_min(scale, df_div(number, df_mul(step, count)));
    sm      = df_mul(sm, scale);
    sn      = df_mul(sn, scale);
    gm      = df_mul(gm, scale);
    gn      = df_mul(gn, scale);
    mass    = df_add(sm, gm);
    count   = df_add(sn, gn);
    vec2 im = vec2(0), ic = vec2(0), si = vec2(0), gi = vec2(0);
    if (df_sub(gm, vec2(1.0000000036274937e-15, -3.627493647833322e-24)).x > 0.0 && df_sub(tc, -vec2(8.0, 0.0)).x > 0.0)
    {
        vec2 factor = vec2(0);
        if (df_sub(tc, -vec2(5.0, 0.0)).x >= 0.0 && df_sub(tc, -vec2(3.0, 0.0)).x < 0.0)
            factor = df_mul(vec2(0.5, 0.0), df_sub(-vec2(3.0, 0.0), tc));
        else if (df_sub(tc, -vec2(8.0, 0.0)).x > 0.0 && df_sub(tc, -vec2(5.0, 0.0)).x < 0.0)
            factor = df_div(df_add(vec2(8.0, 0.0), tc), vec2(3.0, 0.0));
        ic       = df_mul(df_mul(vec2(350000000.0, 0.0), factor), gm);
        im       = df_mul(vec2(9.999999960041972e-13, 3.9958027796366288e-21), ic);
        vec2 sum = df_add(sm, gm);
        if (sum.x > 0.0)
        {
            si = df_mul(df_div(sm, sum), im);
            gi = df_mul(df_div(gm, sum), im);
        }
    }
    if (df_sub(sm, df_mul(vec2(2.0, 0.0), snow_vapor)).x > 0.0 &&
        df_sub(snow_vapor, vec2(1.0000000036274937e-15, -3.627493647833322e-24)).x > 0.0 && sd.x > 0.0)
    {
        vec2 ratio = df_min(vec2(30.0, 0.0), df_div(sm, snow_vapor));
        snow_speed = scalar(
            df_min(vec2(1.5, 0.0),
                   df_add(vec2(1.1000000238418579, -2.3841857821338408e-08),
                          df_mul(df_sub(ratio, vec2(2.0, 0.0)), vec2(0.014000000432133675, -4.3213367433014849e-10)))));
        vec2 fraction = df_min(
                 vec2(0.94999998807907104, 1.1920928910669204e-08),
                 df_add(vec2(0.15000000596046448, -5.9604644830901776e-09),
                        df_mul(df_sub(ratio, vec2(2.0, 0.0)), vec2(0.028000000864267349, -8.6426734866029697e-10)))),
             velocity =
                 df_mul(df_mul(vec2(40.0, 0.0), df_real_power(sd, vec2(0.55000001192092896, -1.1920928910669204e-08))),
                        exponential(df_mul(-vec2(100.0, 0.0), sd)));
        vec2 coordinate   = df_min(vec2(10.0, 0.0),
                                   df_max(vec2(0.10000000149011612, -1.4901161138336505e-09),
                                          df_div(df_mul(-df_mul(cd, vec2(500000.0, 0.0)), velocity),
                                                 df_min(-vec2(0.10000000149011612, -1.4901161138336505e-09), tc)))),
             density_rime = df_mul(
                 df_sub(df_add(vec2(0.050999999046325684, 9.5367431313109208e-10),
                               df_mul(vec2(0.11400000005960464, -5.9604640667565434e-11), coordinate)),
                        df_mul(df_mul(vec2(0.0054999999701976776, 2.9802322068506193e-11), coordinate), coordinate)),
                 vec2(1000.0, 0.0));
        if (df_sub(density_rime, vec2(150.0, 0.0)).x < 0.0)
            fraction = vec2(0);
        vec2 converted = df_mul(fraction, sm);
        sm             = df_sub(sm, converted);
        gm             = df_add(gm, converted);
    }
    sm = df_sub(sm, si);
    gm = df_sub(gm, gi);
    transfer_mass(1, 3, df_div(df_mul(step, im), rho), vec2(0));
    transfer_mass(1, 4, df_div(df_mul(step, sm), rho), vec2(0));
    transfer_mass(1, 5, df_div(df_mul(step, gm), rho), vec2(0));
    heat = df_add(heat, df_div(df_mul(df_mul(step, df_div(vec2(334000.0, 0.0), vec2(1004.0, 0.0))), mass), rho));
    vec2 original = current_number(0);
    store_number(0, df_sub(original, df_div(df_mul(step, count), rho)));
    add_number(2, df_div(ic, rho));
}
void melting(vec2 tw)
{
    vec2 snow = df_mul(q[4], rho), graupel = df_mul(q[5], rho), vapor = q[0],
         tc = df_sub(t, vec2(273.14999389648438, 6.1035156022626325e-06));
    if (tc.x <= 0.0 || (df_sub(snow, vec2(9.999999960041972e-13, 3.9958027796366288e-21)).x <= 0.0 &&
                        df_sub(graupel, vec2(9.999999960041972e-13, 3.9958027796366288e-21)).x <= 0.0))
        return;
    vec2 diffusion    = df_mul(df_mul(vec2(2.1100000594742596e-05, -5.9474259488079793e-13),
                                      df_real_power(df_div(t, vec2(273.14999389648438, 6.1035156022626325e-06)),
                                                    vec2(1.940000057220459, -5.7220459037665705e-08))),
                               df_div(vec2(101325.0, 0.0), p)),
         viscosity    = df_mul(df_add(vec2(1.718000054359436, -5.4359436063577959e-08),
                                      df_mul(vec2(0.0049000000581145287, -5.811452881421264e-11), tc)),
                               vec2(9.9999997473787516e-06, 2.5262124918247909e-13));
    vec2 vent         = root(df_div(rho, viscosity)),
         density_root = root(root(df_div(
             df_div(vec2(101325.0, 0.0), df_mul(vec2(287.04998779296875, 1.2207031261368684e-05), vec2(298.0, 0.0))),
             rho)));
    vec2 conductivity = df_mul(df_mul(df_add(vec2(5.690000057220459, -5.7220458593576495e-08),
                                             df_mul(vec2(0.01679999940097332, 5.9902667895461725e-10), tc)),
                                      vec2(9.9999997473787516e-06, 2.5262124918247909e-13)),
                               vec2(418.93600463867188, -4.6386718963731255e-06)),
         sat          = liquid_sat(p, vec2(273.14999389648438, 6.1035156022626325e-06));
    vec2 deficit      = df_max(vec2(0.0, 0.0), df_sub(sat, vapor)),
         driver       = df_sub(df_mul(tc, conductivity), df_mul(df_mul(vec2(2500000.0, 0.0), diffusion), deficit));
    vec2 sm = vec2(0), gm = vec2(0), nr = vec2(0);
    if (df_sub(snow, vec2(9.999999960041972e-13, 3.9958027796366288e-21)).x > 0.0)
    {
        vec2 st     = df_min(-vec2(0.10000000149011612, -1.4901161138336505e-09), tc),
             second = df_div(snow, vec2(0.068999998271465302, 1.7285347042594879e-09));
        vec2 zeroth = snow_moment(second, st, vec2(0)), first = snow_moment(second, st, vec2(1.0, 0.0)),
             vm = snow_moment(second, st, vec2(1.7749999761581421, 2.3841857821338408e-08));
        sm      = df_min(
            df_div(snow, step),
            df_max(
                vec2(0.0, 0.0),
                df_mul(
                    df_mul(driver,
                           df_div(df_mul(df_mul(vec2(3.1415927410125732, -8.7412573179790343e-08), vec2(4.0, 0.0)),
                                         vec2(0.15000000596046448, -5.9604644830901776e-09)),
                                  vec2(334000.0, 0.0))),
                    df_add(
                        df_mul(vec2(0.86000001430511475, -1.4305114759416426e-08), first),
                        df_mul(df_mul(df_mul(df_mul(df_mul(vec2(0.2800000011920929, -1.1920928688624599e-09),
                                                           root3(vec2(0.63200002908706665, -2.9087066644173376e-08))),
                                                    root(vec2(40.0, 0.0))),
                                             density_root),
                                      vent),
                               vm)))));
        nr = df_add(nr,
                    df_mul(df_mul(df_div(zeroth, snow), sm),
                           df_real_power(vec2(10.0, 0.0),
                                         df_mul(-vec2(0.25, 0.0),
                                                df_sub(tw, vec2(273.14999389648438, 6.1035156022626325e-06))))));
    }
    if (df_sub(graupel, vec2(9.999999960041972e-13, 3.9958027796366288e-21)).x > 0.0)
    {
        vec2 n0 = df_real_power(
            vec2(10.0, 0.0),
            df_min(
                vec2(6.0, 0.0),
                df_max(vec2(2.0, 0.0),
                       df_add(vec2(3.0, 0.0),
                              df_mul(df_div(vec2(2.0, 0.0), vec2(7.0, 0.0)),
                                     df_add(df_div(logarithm(df_max(
                                                       vec2(9.9999997171806854e-10, 2.8281931525706872e-17), graupel)),
                                                   logarithm(vec2(10.0, 0.0))),
                                            vec2(8.0, 0.0)))))));
        vec2 exponential_lam = df_real_power(
                 df_div(df_mul(n0,
                               df_div(df_mul(vec2(3.1415927410125732, -8.7412573179790343e-08), vec2(400.0, 0.0)),
                                      vec2(6.0, 0.0))),
                        graupel),
                 vec2(0.25, 0.0)),
             lam = df_mul(exponential_lam, root3(vec2(6.0, 0.0))), inv = df_div(vec2(1.0, 0.0), lam);
        vec2 intercept = df_mul(df_div(n0, exponential_lam), lam), number = df_div(intercept, lam),
             melt_intercept = intercept;
        if (df_sub(df_mul(graupel, number), vec2(9.9999997473787516e-05, 2.5262124884366591e-12)).x < 0.0)
            melt_intercept = df_mul(df_div(vec2(9.9999997473787516e-05, 2.5262124884366591e-12), graupel), lam);
        vec2 gamma_v = coeff(14862121),
             first   = df_mul(df_div(df_mul(df_mul(vec2(3.1415927410125732, -8.7412573179790343e-08), vec2(4.0, 0.0)),
                                            vec2(0.5, 0.0)),
                                     vec2(334000.0, 0.0)),
                              vec2(0.86000001430511475, -1.4305114759416426e-08)),
             second =
                 df_mul(df_mul(df_mul(df_mul(df_div(df_mul(df_mul(vec2(3.1415927410125732, -8.7412573179790343e-08),
                                                                  vec2(4.0, 0.0)),
                                                           vec2(0.5, 0.0)),
                                                    vec2(334000.0, 0.0)),
                                             vec2(0.2800000011920929, -1.1920928688624599e-09)),
                                      root3(vec2(0.63200002908706665, -2.9087066644173376e-08))),
                               root(vec2(442.0, 0.0))),
                        gamma_v);
        gm = df_min(
            df_div(graupel, step),
            df_max(vec2(0.0, 0.0),
                   df_mul(df_mul(driver, melt_intercept),
                          df_add(df_mul(df_mul(first, inv), inv),
                                 df_mul(df_mul(df_mul(second, density_root), vent),
                                        df_real_power(inv, vec2(2.9449999332427979, 6.6757201988565384e-08)))))));
        nr = df_add(nr,
                    df_mul(df_div(df_mul(gm, number), graupel),
                           df_real_power(vec2(10.0, 0.0),
                                         df_mul(-vec2(0.33000001311302185, -1.3113021835042815e-08),
                                                df_sub(tw, vec2(273.14999389648438, 6.1035156022626325e-06))))));
    }
    transfer_mass(4, 2, df_div(df_mul(step, sm), rho), vec2(0));
    transfer_mass(5, 2, df_div(df_mul(step, gm), rho), vec2(0));
    heat =
        df_sub(heat,
               df_div(df_mul(df_div(df_mul(step, vec2(334000.0, 0.0)),
                                    df_mul(vec2(1004.0, 0.0),
                                           df_add(vec2(1.0, 0.0),
                                                  df_mul(vec2(0.88700002431869507, -2.4318695057701234e-08), vapor)))),
                             df_add(sm, gm)),
                      rho));
    add_number(1, df_div(nr, rho));
}
vec2 collection(vec2 tw)
{
    vec2 r = df_mul(q[2], rho), s = df_mul(q[4], rho), g = df_mul(q[5], rho), number = df_mul(current_number(1), rho);
    vec2 rr = vec2(0), sr = vec2(0), gr = vec2(0), nr = vec2(0);
    vec2 intercept = vec2(10000000000.0, 0.0);
    if (df_sub(r, vec2(9.9999999747524271e-07, 2.5247572468697606e-15)).x > 0.0)
    {
        vec2 lam = root3(
            df_div(df_mul(df_mul(vec2(3.1415927410125732, -8.7412573179790343e-08), vec2(1000.0, 0.0)), number), r));
        intercept = df_mul(df_div(r, df_mul(vec2(3.1415927410125732, -8.7412573179790343e-08), vec2(1000.0, 0.0))),
                           df_power(lam, 4));
    }
    int ri = axis_index(r, 14861891, 37), rn = axis_index(intercept, 14861928, 37);
    if (df_sub(r, vec2(9.9999999747524271e-07, 2.5247572468697606e-15)).x >= 0.0 &&
        df_sub(s, vec2(9.9999999747524271e-07, 2.5247572468697606e-15)).x >= 0.0)
    {
        vec2 coordinate = df_div(df_sub(df_sub(tw, vec2(273.14999389648438, 6.1035156022626325e-06)), vec2(2.5, 0.0)),
                                 vec2(5.0, 0.0));
        int  truncated = int(scalar(coordinate)) - 1, ti = clamp(-truncated, 1, 9) - 1;
        int  index = ((axis_index(s, 14861891, 37) * 9 + ti) * 37 + rn) * 37 + ri;
        rs_index   = float(index);
        vec2 v0 = coeff(9370805 + 12 * index), v1 = coeff(9370805 + 12 * index + 1),
             v3 = coeff(9370805 + 12 * index + 3), v4 = coeff(9370805 + 12 * index + 4),
             v5 = coeff(9370805 + 12 * index + 5), v6 = coeff(9370805 + 12 * index + 6);
        if (df_sub(tw, vec2(273.14999389648438, 6.1035156022626325e-06)).x < 0.0)
        {
            rr = df_sub(df_sub(df_sub(df_sub(rr, v3), v6), v1), v4);
            sr = df_sub(df_sub(df_add(df_add(sr, v3), v6), v0), v5);
            gr = df_add(df_add(df_add(df_add(gr, v1), v4), v0), v5);
            nr = df_sub(nr,
                        df_add(df_add(coeff(9370805 + 12 * index + 8), coeff(9370805 + 12 * index + 9)),
                               df_add(coeff(9370805 + 12 * index + 10), coeff(9370805 + 12 * index + 11))));
        }
        else
        {
            sr = df_add(df_add(df_sub(df_sub(sr, v0), v5), v3), v6);
            rr = df_sub(rr, sr);
        }
    }
    if (df_sub(r, vec2(9.9999999747524271e-07, 2.5247572468697606e-15)).x >= 0.0 &&
        df_sub(g, vec2(9.9999999747524271e-07, 2.5247572468697606e-15)).x >= 0.0)
    {
        vec2 g0 = df_real_power(
            vec2(10.0, 0.0),
            df_min(vec2(6.0, 0.0),
                   df_max(vec2(2.0, 0.0),
                          df_add(vec2(3.0, 0.0),
                                 df_mul(df_div(vec2(2.0, 0.0), vec2(7.0, 0.0)),
                                        df_add(df_div(logarithm(df_max(
                                                          vec2(9.9999997171806854e-10, 2.8281931525706872e-17), g)),
                                                      logarithm(vec2(10.0, 0.0))),
                                               vec2(8.0, 0.0)))))));
        int index = ((axis_index(g0, 14861965, 37) * 37 + axis_index(g, 14861891, 37)) * 37 + rn) * 37 + ri;
        rg_index  = float(index);
        vec2 v0 = coeff(5 * index), v1 = coeff(5 * index + 1), v2 = coeff(5 * index + 2), v3 = coeff(5 * index + 3),
             v4 = coeff(5 * index + 4);
        if (df_sub(tw, vec2(273.14999389648438, 6.1035156022626325e-06)).x < 0.0)
        {
            vec2 frozen = df_min(df_div(r, step), df_add(v1, v2));
            rr          = df_sub(rr, frozen);
            gr          = df_add(gr, frozen);
            nr          = df_sub(nr, df_min(df_div(number, step), df_add(v3, v4)));
        }
        else
        {
            vec2 melted = df_min(df_div(g, step), v0);
            rr          = df_add(rr, melted);
            gr          = df_sub(gr, melted);
            nr          = df_add(nr, df_mul(vec2(1.5, 0.0), v4));
        }
    }
    vec2 scale = vec2(1, 0);
    if (rr.x < 0.0)
        scale = df_min(scale, df_div(r, df_mul(-rr, step)));
    if (sr.x < 0.0)
        scale = df_min(scale, df_div(s, df_mul(-sr, step)));
    if (gr.x < 0.0)
        scale = df_min(scale, df_div(g, df_mul(-gr, step)));
    rr = df_mul(rr, scale);
    sr = df_mul(sr, scale);
    gr = df_mul(gr, scale);
    nr = df_max(df_div(-number, step), df_mul(nr, scale));
    vec2 mass_rates[3];
    mass_rates[0] = rr;
    mass_rates[1] = sr;
    mass_rates[2] = gr;
    int species[3];
    species[0]    = 2;
    species[1]    = 4;
    species[2]    = 5;
    vec2 positive = df_add(df_add(df_max(vec2(0.0, 0.0), rr), df_max(vec2(0.0, 0.0), sr)), df_max(vec2(0.0, 0.0), gr)),
         before   = q[2];
    for (int donor = 0; donor < 3; ++donor)
    {
        if (mass_rates[donor].x >= 0.0)
            continue;
        vec2 remaining   = df_min(q[species[donor]], df_div(df_mul(step, -mass_rates[donor]), rho)),
             denominator = positive;
        for (int receiver = 0; receiver < 3; ++receiver)
        {
            if (mass_rates[receiver].x <= 0.0)
                continue;
            vec2 amount = df_div(df_mul(remaining, mass_rates[receiver]), denominator);
            transfer_mass(species[donor], species[receiver], amount, vec2(0));
            remaining   = df_sub(remaining, amount);
            denominator = df_sub(denominator, mass_rates[receiver]);
        }
    }
    vec2 after     = q[2];
    heat           = df_sub(heat, df_mul(df_div(vec2(334000.0, 0.0), vec2(1004.0, 0.0)), df_sub(after, before)));
    vec2 current   = current_number(1);
    vec2 candidate = df_max(vec2(0.0, 0.0), df_add(current, df_div(df_mul(step, nr), rho)));
    store_number(1, candidate);
    return candidate;
}
vec2 efficiency(vec2 diameter, vec2 aerosol, vec2 viscosity, int species)
{
    vec2 velocity;
    if (species == 0)
        velocity = df_sub(df_add(df_sub(df_add(-vec2(0.10209999978542328, 2.1457671750546581e-10),
                                               df_mul(vec2(4932.0, 0.0), diameter)),
                                        df_mul(vec2(955100.0, 0.0), df_power(diameter, 2))),
                                 df_mul(vec2(79340000.0, 0.0), df_power(diameter, 3))),
                          df_mul(vec2(2361999872.0, 128.0), df_power(diameter, 4)));
    else if (species == 1)
        velocity = df_mul(vec2(40.0, 0.0), df_real_power(diameter, vec2(0.55000001192092896, -1.1920928910669204e-08)));
    else
        velocity = df_mul(vec2(442.0, 0.0), df_real_power(diameter, vec2(0.88999998569488525, 1.4305114759416426e-08)));
    if (!finite_pair(velocity) || velocity.x <= 0.0)
    {
        failed = true;
        return vec2(0);
    }
    vec2 slip = df_add(
        vec2(1.0, 0.0),
        df_mul(df_div(df_mul(vec2(2.0, 0.0), vec2(2.5600000341796658e-08, -3.4179665724379885e-16)), aerosol),
               df_add(vec2(1.2569999694824219, 3.0517578020194946e-08),
                      df_mul(vec2(0.40000000596046448, -5.960464455334602e-09),
                             exponential(df_div(df_mul(-vec2(0.55000001192092896, -1.1920928910669204e-08), aerosol),
                                                vec2(2.5600000341796658e-08, -3.4179665724379885e-16)))))));
    vec2 diffusion = df_div(
        df_mul(df_mul(vec2(1.3806503145937019e-23, -1.4593701827499632e-31), t), slip),
        df_mul(df_mul(df_mul(vec2(3.0, 0.0), vec2(3.1415927410125732, -8.7412573179790343e-08)), viscosity), aerosol));
    vec2 re        = df_div(df_mul(df_mul(df_mul(vec2(0.5, 0.0), rho), diameter), velocity), viscosity),
         sc        = df_div(viscosity, df_mul(rho, diffusion)),
         st        = df_div(df_mul(df_mul(df_mul(aerosol, aerosol), velocity), vec2(1000.0, 0.0)),
                            df_mul(df_mul(vec2(9.0, 0.0), viscosity), diameter));
    vec2 a         = df_add(vec2(1.0, 0.0), logarithm(df_add(vec2(1.0, 0.0), re))),
         threshold = df_div(df_add(vec2(1.2000000476837158, -4.7683715864721421e-08), df_div(a, vec2(12.0, 0.0))),
                            df_add(vec2(1.0, 0.0), a));
    vec2 value     = df_add(
        df_mul(df_div(vec2(4.0, 0.0), df_mul(re, sc)),
               df_add(df_add(vec2(1.0, 0.0),
                             df_mul(df_mul(vec2(0.40000000596046448, -5.960464455334602e-09), root(re)), root3(sc))),
                      df_mul(df_mul(vec2(0.15999999642372131, 3.5762786898541066e-09), root(re)), root(sc)))),
        df_mul(df_div(df_mul(vec2(4.0, 0.0), aerosol), diameter),
               df_add(vec2(0.019999999552965164, 4.4703483623176332e-10),
                      df_mul(df_div(aerosol, diameter), df_add(vec2(1.0, 0.0), df_mul(vec2(2.0, 0.0), root(re)))))));
    if (df_sub(st, threshold).x > 0.0)
        value = df_add(
            value,
            df_real_power(df_div(df_sub(st, threshold),
                                 df_add(df_sub(st, threshold), vec2(0.66666698455810547, 1.5441894540835222e-08))),
                          vec2(1.5, 0.0)));
    return df_min(vec2(1.0, 0.0), df_max(vec2(9.9999997473787516e-06, 2.5262124918247909e-13), value));
}
void scavenging(vec2 rain_number)
{
    vec2 tc = df_sub(t, vec2(273.14999389648438, 6.1035156022626325e-06)), viscosity;
    if (tc.x >= 0.0)
        viscosity = df_mul(df_add(vec2(1.718000054359436, -5.4359436063577959e-08),
                                  df_mul(vec2(0.0049000000581145287, -5.811452881421264e-11), tc)),
                           vec2(9.9999997473787516e-06, 2.5262124918247909e-13));
    else
        viscosity = df_mul(df_sub(df_add(vec2(1.718000054359436, -5.4359436063577959e-08),
                                         df_mul(vec2(0.0049000000581145287, -5.811452881421264e-11), tc)),
                                  df_mul(df_mul(vec2(1.2000000424450263e-05, -4.2445026307699404e-13), tc), tc)),
                           vec2(9.9999997473787516e-06, 2.5262124918247909e-13));
    vec2 r = df_mul(q[2], rho), s = df_mul(q[4], rho), g = df_mul(q[5], rho), nr = df_mul(rain_number, rho);
    vec2 nw = current_number(3), na = current_number(4), nwc = df_mul(nw, rho), nac = df_mul(na, rho);
    vec2 rhof = root(df_div(
             df_div(vec2(101325.0, 0.0), df_mul(vec2(287.04998779296875, 1.2207031261368684e-05), vec2(298.0, 0.0))),
             rho)),
         wr = vec2(0), ir = vec2(0);
    if (df_sub(r, vec2(9.999999960041972e-13, 3.9958027796366288e-21)).x > 0.0 &&
        df_sub(nr, vec2(9.9999999747524271e-07, 2.5247572468697606e-15)).x > 0.0)
    {
        vec2 lam = root3(
                 df_div(df_mul(df_mul(vec2(3.1415927410125732, -8.7412573179790343e-08), vec2(1000.0, 0.0)), nr), r)),
             diameter = df_div(vec2(3.6719999313354492, 6.8664550934016688e-08), lam);
        if (df_sub(diameter, vec2(4.9999998736893758e-05, 1.2631062442183295e-12)).x > 0.0)
        {
            vec2 rate_factor = df_div(
                df_mul(df_mul(df_mul(df_mul(df_mul(df_mul(rhof, vec2(3.1415927410125732, -8.7412573179790343e-08)),
                                                   vec2(0.25, 0.0)),
                                            vec2(4854.0, 0.0)),
                                     vec2(6.0, 0.0)),
                              nr),
                       lam),
                df_power(df_add(lam, vec2(195.0, 0.0)), 4));
            wr = df_add(
                wr,
                df_mul(df_mul(rate_factor, nwc),
                       efficiency(diameter, vec2(3.9999999756901161e-08, 2.4309883967378842e-16), viscosity, 0)));
            ir = df_add(
                ir,
                df_mul(df_mul(rate_factor, nac),
                       efficiency(diameter, vec2(8.0000000934887794e-07, -9.3488779746657945e-15), viscosity, 0)));
        }
    }
    if (df_sub(s, vec2(9.9999999747524271e-07, 2.5247572468697606e-15)).x > 0.0)
    {
        vec2 st     = df_min(-vec2(0.10000000149011612, -1.4901161138336505e-09), tc),
             second = df_div(s, vec2(0.068999998271465302, 1.7285347042594879e-09)),
             third = snow_moment(second, st, vec2(3.0, 0.0)), diameter = df_div(third, second),
             moment = snow_moment(second, st, vec2(2.5499999523162842, 4.7683715642676816e-08));
        vec2 rate_factor =
            df_mul(df_mul(df_mul(df_mul(rhof, vec2(3.1415927410125732, -8.7412573179790343e-08)), vec2(0.25, 0.0)),
                          vec2(40.0, 0.0)),
                   moment);
        wr = df_add(wr,
                    df_mul(df_mul(rate_factor, nwc),
                           efficiency(diameter, vec2(3.9999999756901161e-08, 2.4309883967378842e-16), viscosity, 1)));
        ir = df_add(ir,
                    df_mul(df_mul(rate_factor, nac),
                           efficiency(diameter, vec2(8.0000000934887794e-07, -9.3488779746657945e-15), viscosity, 1)));
    }
    if (df_sub(g, vec2(9.9999999747524271e-07, 2.5247572468697606e-15)).x > 0.0)
    {
        vec2 seed = df_real_power(
                 vec2(10.0, 0.0),
                 df_min(
                     vec2(6.0, 0.0),
                     df_max(vec2(2.0, 0.0),
                            df_add(vec2(3.0, 0.0),
                                   df_mul(df_div(vec2(2.0, 0.0), vec2(7.0, 0.0)),
                                          df_add(df_div(logarithm(df_max(
                                                            vec2(9.9999997171806854e-10, 2.8281931525706872e-17), g)),
                                                        logarithm(vec2(10.0, 0.0))),
                                                 vec2(8.0, 0.0))))))),
             am  = df_div(df_mul(vec2(3.1415927410125732, -8.7412573179790343e-08), vec2(400.0, 0.0)), vec2(6.0, 0.0));
        vec2 lam = df_mul(df_real_power(df_div(df_mul(seed, am), g), vec2(0.25, 0.0)), root3(vec2(6.0, 0.0))),
             number = df_div(df_mul(g, df_power(lam, 3)), df_mul(vec2(6.0, 0.0), am)), intercept = df_mul(number, lam),
             diameter = df_div(vec2(4.0, 0.0), lam), gamma_v = coeff(14862121 + 1);
        vec2 rate_factor = df_div(
            df_mul(
                df_mul(df_mul(df_mul(df_mul(rhof, vec2(3.1415927410125732, -8.7412573179790343e-08)), vec2(0.25, 0.0)),
                              vec2(442.0, 0.0)),
                       gamma_v),
                intercept),
            df_real_power(lam, vec2(3.8900001049041748, -1.0490417468034252e-07)));
        wr = df_add(wr,
                    df_mul(df_mul(rate_factor, nwc),
                           efficiency(diameter, vec2(3.9999999756901161e-08, 2.4309883967378842e-16), viscosity, 2)));
        ir = df_add(ir,
                    df_mul(df_mul(rate_factor, nac),
                           efficiency(diameter, vec2(8.0000000934887794e-07, -9.3488779746657945e-15), viscosity, 2)));
    }
    if (!finite_pair(wr) || !finite_pair(ir) || wr.x < 0.0 || ir.x < 0.0)
    {
        failed = true;
        return;
    }
    wr = df_min(df_div(nw, step), df_div(wr, rho));
    ir = df_min(df_div(na, step), df_div(ir, rho));
    store_number(3, df_max(vec2(0.0, 0.0), df_sub(nw, df_mul(step, wr))));
    store_number(4, df_max(vec2(0.0, 0.0), df_sub(na, df_mul(step, ir))));
}
