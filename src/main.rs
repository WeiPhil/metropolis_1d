mod functions;
mod metropolis;
mod pcg32;

use std::rc::Rc;

use crate::metropolis::Metropolis;
use egui::{plot::*};
use functions::*;
use metropolis::Function;

use egui_glow::glow::HasContext;

fn create_display(
    event_loop: &glutin::event_loop::EventLoop<()>,
) -> (
    glutin::WindowedContext<glutin::PossiblyCurrent>,
    egui_glow::glow::Context,
) {
    let window_builder = glutin::window::WindowBuilder::new()
        .with_resizable(true)
        .with_inner_size(glutin::dpi::LogicalSize::<f32> {
            width: 800.0,
            height: 600.0,
        })
        .with_title("1D Metropolis Sampling");

    let gl_window = unsafe {
        glutin::ContextBuilder::new()
            .with_depth_buffer(0)
            .with_srgb(true)
            .with_stencil_buffer(0)
            .with_vsync(true)
            .build_windowed(window_builder, event_loop)
            .unwrap()
            .make_current()
            .unwrap()
    };

    let gl = unsafe {
        egui_glow::glow::Context::from_loader_function(|s| gl_window.get_proc_address(s))
    };

    (gl_window, gl)
}

/// A helper function to handle responses in the ui. It sets the update_parent to true
/// if the widget was either dragged or we lost focus (finished typing, clicked somewhere else)
#[inline(always)]
pub fn drag_or_lost_focus(response: egui::Response) -> bool {
    (response.dragged() && response.changed()) || response.lost_focus()
}

struct MetropolisApp {
    ref_function: fn(f64) -> f64,
    expected_value_technique: bool,
    seed: u64,
    small_mutate_prob: f32,
    burn_in_samples: usize,
    metropolis_samples: usize,
    num_bins: usize,
    f_and_norm: (Box<Function<f32>>, f32),
    metropolis: Metropolis,
    samples_distribution: Vec<f32>,
}

impl MetropolisApp {
    pub fn new() -> MetropolisApp {
        // Initialise all the required quantities that can be modified
        let ref_function = shifted_square_f64 as fn(f64) -> f64;
        let expected_value_technique = false;
        let seed = 0;
        let small_mutate_prob = 0.5;
        let burn_in_samples = 100;
        let metropolis_samples = 10000;
        let num_bins = 50;
        let f_and_norm = shifted_square_and_norm();
        let metropolis = Metropolis::gen_sample_sequence(
            seed,
            &f_and_norm,
            metropolis_samples,
            burn_in_samples,
            small_mutate_prob,
            expected_value_technique,
        );
        let samples_distribution = metropolis.sample_distribution(num_bins);

        MetropolisApp {
            ref_function,
            expected_value_technique,
            seed,
            small_mutate_prob,
            burn_in_samples,
            metropolis_samples,
            num_bins,
            f_and_norm,
            metropolis,
            samples_distribution,
        }
    }

    pub fn update_ui(
        &mut self,
        egui: &mut egui_glow::EguiGlow,
        window: &glutin::window::Window,
    ) -> bool {
        let mut should_update = false;

        let need_redraw = egui.run(window, |ctx| {

                egui::SidePanel::left("left pannel")
                .min_width(200.0)
                .default_width(300.0)
                .show(ctx, |ui| {
                    ui.vertical_centered_justified(|ui| {
                        ui.add_space(5.0);
                        ui.heading("1D Metropolis Sampling");

                        ui.add_space(5.0);
                        ui.separator();
                        ui.add_space(5.0);

                        if ui
                            .checkbox(&mut self.expected_value_technique, "expected value technique")
                            .changed()
                        {
                            should_update = true;
                        };

                        ui.add_space(5.0);
                        ui.separator();
                        ui.add_space(5.0);

                        ui.vertical_centered_justified(|ui| {
                            let mut change_function = FunctionType::None;
                            egui::ComboBox::from_id_source("change_function")
                                .selected_text("Change Function")
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(
                                        &mut change_function,
                                        FunctionType::ShiftedSquare,
                                        "Shifted Square",
                                    );
                                    ui.selectable_value(
                                        &mut change_function,
                                        FunctionType::Sinus,
                                        "Sinus",
                                    );
                                });

                            match change_function {
                                FunctionType::ShiftedSquare => {
                                    self.f_and_norm = shifted_square_and_norm();
                                    self.ref_function = shifted_square_f64 as fn(f64) -> f64;
                                    should_update = true;
                                }
                                FunctionType::Sinus => {
                                    self.f_and_norm = sinus_and_norm();
                                    self.ref_function = sinus_f64 as fn(f64) -> f64;
                                    should_update = true;
                                }
                                FunctionType::None => (),
                            }
                        });

                        ui.add_space(5.0);
                        ui.separator();
                        ui.add_space(5.0);

                        ui.horizontal(|ui| {
                            ui.label("rand seed : ");
                            if drag_or_lost_focus(
                                ui.add(egui::DragValue::new(&mut self.seed).clamp_range(0..=u64::MAX)),
                            ) {
                                should_update = true;
                            };
                        });

                        ui.horizontal(|ui| {
                            ui.label("burn-in samples : ").on_hover_text("How many samples are used to initialise the markov-chain and remove the start-up bias.");
                            if drag_or_lost_focus(
                                ui.add(
                                    egui::DragValue::new(&mut self.burn_in_samples)
                                        .clamp_range(0..=usize::MAX),
                                ),
                            ) {
                                should_update = true;
                            };
                        });

                        ui.horizontal(|ui| {
                            ui.label("metropolis samples: ").on_hover_text("How many samples are generated via metropolis sampling");
                            if drag_or_lost_focus(
                                ui.add(
                                    egui::DragValue::new(&mut self.metropolis_samples)
                                        .clamp_range(1..=usize::MAX),
                                ),
                            ) {
                                should_update = true;
                            };
                        });
                        ui.horizontal(|ui| {
                            ui.label("num bins : ").on_hover_text("The number of bins used to estimate the generated sample density");
                            if drag_or_lost_focus(ui.add(
                                egui::DragValue::new(&mut self.num_bins).clamp_range(1..=usize::MAX),
                            )) {
                                should_update = true;
                            };
                        });

                        ui.horizontal(|ui| {
                            ui.label("small mutation prob : ").on_hover_text("The probability of performing a small mutation (instead of a large mutation)");
                            if drag_or_lost_focus(
                                ui.add(
                                    egui::DragValue::new(&mut self.small_mutate_prob)
                                        .speed(0.05)
                                        .clamp_range(0.0f32..=1.0),
                                ),
                            ) {
                                should_update = true;
                            };
                        });

                    });
                });

                self.update_plot(ctx)          
            });

            if should_update {
                self.metropolis = Metropolis::gen_sample_sequence(
                    self.seed,
                    &self.f_and_norm,
                    self.metropolis_samples,
                    self.burn_in_samples,
                    self.small_mutate_prob,
                    self.expected_value_technique,
                );
                self.samples_distribution = self.metropolis.sample_distribution(self.num_bins);
            }

            need_redraw
    }

    fn objective_function(&self) -> Line{
        Line::new(Values::from_explicit_callback(self.ref_function, .., 1024))
        .color(egui::Color32::from_rgb(100, 200, 100))
        .name("reference")
    }

    fn metropolis_function(&self) -> BarChart{
        let bars  = BarChart::new(
            self.samples_distribution
                .iter()
                .zip(0..self.num_bins)
                .map(|(y, x)| {
                        Bar::new((x as f64 +0.5)/ self.num_bins as f64, *y as f64)
                }).collect(),
        )
        .width(1.0/self.num_bins as f64)
        .color(egui::Color32::from_rgb(200, 100, 100))
        .name("sampled");

        bars
    }

    pub fn update_plot(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {

            let plot = Plot::new("lines_demo")
            .legend(Legend::default())
            .allow_zoom(true)
            .data_aspect(1.0);

            plot.show(ui, |plot_ui| {
                plot_ui.line(self.objective_function());
                plot_ui.bar_chart(self.metropolis_function());
            });
        });
    }
}

fn main() {
    let event_loop = glutin::event_loop::EventLoop::with_user_event();

    // Setting up egui
    let (gl_window, gl) = create_display(&event_loop);
    let gl = Rc::new(gl);

    let mut egui = egui_glow::EguiGlow::new(&gl_window.window(), gl.clone());

    let mut app = MetropolisApp::new();

    event_loop.run(move |event, _, control_flow| {
        let mut update_and_draw_gui = || {
            *control_flow = if app.update_ui(&mut egui, gl_window.window()) {
                gl_window.window().request_redraw();
                glutin::event_loop::ControlFlow::Poll
            } else {
                glutin::event_loop::ControlFlow::Wait
            };

            {
                unsafe {
                    gl.clear_color(0.1, 0.1, 0.1, 1.0);
                    gl.clear(egui_glow::glow::COLOR_BUFFER_BIT);
                }
                // Everything here is drawn before egui
                egui.paint(gl_window.window());
                // Everything here is drawn on top of egui
                gl_window.swap_buffers().unwrap();
            }
        };

        match event {
            // Platform-dependent event handlers to workaround a winit bug
            // See: https://github.com/rust-windowing/winit/issues/987
            // See: https://github.com/rust-windowing/winit/issues/1619
            glutin::event::Event::RedrawEventsCleared if cfg!(windows) => update_and_draw_gui(),
            glutin::event::Event::RedrawRequested(_) if !cfg!(windows) => update_and_draw_gui(),

            glutin::event::Event::WindowEvent { event, .. } => {
                use glutin::event::WindowEvent;
                if matches!(event, WindowEvent::CloseRequested | WindowEvent::Destroyed) {
                    *control_flow = glutin::event_loop::ControlFlow::Exit;
                }

                if let glutin::event::WindowEvent::Resized(physical_size) = &event {
                    gl_window.resize(*physical_size);
                } else if let glutin::event::WindowEvent::ScaleFactorChanged {
                    new_inner_size,
                    ..
                } = &event
                {
                    gl_window.resize(**new_inner_size);
                }

                egui.on_event(&event);

                gl_window.window().request_redraw(); // TODO: ask egui if the events warrants a repaint instead
            }

            _ => (),
        }
    });
}
