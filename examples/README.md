ITP2V是pv和title 拼接的版本

ITP3V_capture是PV和title分开encode的版本，但是这个版本的问题是对比学习loss的
代码有问题

ITP3V_capture_v2是改了正确的对比学习的版本，但是效果好像有点低


ITP3V_capture_v2_dymask是在上面的版本基础上，增加了mask任务的动态权重


ITP3V_capture_v3是调整了ITP3V_capture的对比学习loss代码
不用n_views的方式，两两pair，还是之前的代码











