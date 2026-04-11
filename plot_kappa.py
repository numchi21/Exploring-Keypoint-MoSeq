import keypoint_moseq as kpms
import numpy as np
fig, durations = kpms.plot_kappa_scan(np.logspace(3,7,5), 'kpms_project/kpms_project', 'kappa_scan')
fig.savefig('kpms_project/kappa_scan_plot.png', dpi=150, bbox_inches='tight')
print('Durations:', durations)
