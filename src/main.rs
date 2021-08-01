mod functions;
mod metropolis;
mod pcg32;

use crate::metropolis::Metropolis;
use egui::plot::*;
use functions::*;
use glium::glutin;

fn create_display(event_loop: &glutin::event_loop::EventLoop<()>) -> glium::Display {
    let window_builder = glutin::window::WindowBuilder::new()
        .with_resizable(true)
        .with_inner_size(glutin::dpi::LogicalSize {
            width: 800.0,
            height: 600.0,
        })
        .with_title("1D Metropolis Sampling");

    let context_builder = glutin::ContextBuilder::new()
        .with_depth_buffer(0)
        .with_srgb(true)
        .with_stencil_buffer(0)
        .with_vsync(true);

    glium::Display::new(window_builder, context_builder, event_loop).unwrap()
}

/// A helper function to handle responses in the ui. It sets the update_parent to true
/// if the widget was either dragged or we lost focus (finished typing, clicked somewhere else)
#[inline(always)]
pub fn drag_or_lost_focus(response: egui::Response) -> bool {
    (response.dragged() && response.changed()) || response.lost_focus()
}

fn main() {
    let event_loop = glutin::event_loop::EventLoop::with_user_event();
    let display = create_display(&event_loop);

    let mut egui = egui_glium::EguiGlium::new(&display);

    // Initialise all the required quantities that can be modified
    let mut ref_function = shifted_square_f64 as fn(f64) -> f64;
    let mut expected_value_technique = false;
    let mut seed = 0;
    let mut small_mutate_prob = 0.5;
    let mut burn_in_samples = 100;
    let mut metropolis_samples = 10000;
    let mut num_bins = 50;
    let mut f_and_norm = shifted_square_and_norm();
    let mut metropolis = Metropolis::gen_sample_sequence(
        seed,
        &f_and_norm,
        metropolis_samples,
        burn_in_samples,
        small_mutate_prob,
        expected_value_technique,
    );
    let mut samples_distribution = metropolis.sample_distribution(num_bins);

    event_loop.run(move |event, _, control_flow| {
        let mut redraw = || {
            egui.begin_frame(&display);

            let mut quit = false;

            let mut should_update = false;

            egui::SidePanel::left("left pannel")
                .min_width(200.0)
                .default_width(300.0)
                .show(egui.ctx(), |ui| {
                    ui.vertical_centered_justified(|ui| {
                        ui.add_space(5.0);
                        ui.heading("1D Metropolis Sampling");

                        ui.add_space(5.0);
                        ui.separator();
                        ui.add_space(5.0);

                        if ui
                            .checkbox(&mut expected_value_technique, "expected value technique")
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
                                    f_and_norm = shifted_square_and_norm();
                                    ref_function = shifted_square_f64 as fn(f64) -> f64;
                                    should_update = true;
                                }
                                FunctionType::Sinus => {
                                    f_and_norm = sinus_and_norm();
                                    ref_function = sinus_f64 as fn(f64) -> f64;
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
                                ui.add(egui::DragValue::new(&mut seed).clamp_range(0..=u64::MAX)),
                            ) {
                                should_update = true;
                            };
                        });

                        ui.horizontal(|ui| {
                            ui.label("burn-in samples : ").on_hover_text("How many samples are used to initialise the markov-chain and remove the start-up bias.");
                            if drag_or_lost_focus(
                                ui.add(
                                    egui::DragValue::new(&mut burn_in_samples)
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
                                    egui::DragValue::new(&mut metropolis_samples)
                                        .clamp_range(1..=usize::MAX),
                                ),
                            ) {
                                should_update = true;
                            };
                        });
                        ui.horizontal(|ui| {
                            ui.label("num bins : ").on_hover_text("The number of bins used to estimate the generated sample density");
                            if drag_or_lost_focus(ui.add(
                                egui::DragValue::new(&mut num_bins).clamp_range(1..=usize::MAX),
                            )) {
                                should_update = true;
                            };
                        });

                        ui.horizontal(|ui| {
                            ui.label("small mutation prob : ").on_hover_text("The probability of performing a small mutation (instead of a large mutation)");
                            if drag_or_lost_focus(
                                ui.add(
                                    egui::DragValue::new(&mut small_mutate_prob)
                                        .speed(0.05)
                                        .clamp_range(0.0..=1.0),
                                ),
                            ) {
                                should_update = true;
                            };
                        });

                        ui.separator();
                        if ui.button("Quit").clicked() {
                            quit = true;
                        }
                    });
                });

            if should_update {
                metropolis = Metropolis::gen_sample_sequence(
                    seed,
                    &f_and_norm,
                    metropolis_samples,
                    burn_in_samples,
                    small_mutate_prob,
                    expected_value_technique,
                );
                samples_distribution = metropolis.sample_distribution(num_bins);
            }

            let objective_function_line =
                Line::new(Values::from_explicit_callback(ref_function, .., 1024))
                    .color(egui::Color32::from_rgb(100, 200, 100))
                    .name("reference");
            let metropolis_line = Line::new(Values::from_values_iter(
                samples_distribution
                    .iter()
                    .zip(0..num_bins)
                    .flat_map(|(y, x)| {
                        vec![
                            Value::new(x as f32 / num_bins as f32, *y),
                            Value::new((x + 1) as f32 / num_bins as f32, *y),
                        ]
                        .into_iter()
                    }),
            ))
            .color(egui::Color32::from_rgb(200, 100, 100))
            .name("sampled");

            egui::CentralPanel::default().show(egui.ctx(), |ui| {
                let plot = Plot::new("lines_demo")
                    .line(objective_function_line)
                    .line(metropolis_line)
                    .legend(Legend::default())
                    .data_aspect(1.0);

                ui.add(plot);
            });

            let (needs_repaint, shapes) = egui.end_frame(&display);

            *control_flow = if quit {
                glutin::event_loop::ControlFlow::Exit
            } else if needs_repaint {
                display.gl_window().window().request_redraw();
                glutin::event_loop::ControlFlow::Poll
            } else {
                glutin::event_loop::ControlFlow::Wait
            };

            {
                use glium::Surface as _;
                let mut target = display.draw();

                let clear_color = egui::Rgba::from_gray(0.1);
                target.clear_color(
                    clear_color[0],
                    clear_color[1],
                    clear_color[2],
                    clear_color[3],
                );

                // draw things behind egui here

                egui.paint(&display, &mut target, shapes);

                // draw things on top of egui here

                target.finish().unwrap();
            }
        };

        match event {
            // Platform-dependent event handlers to workaround a winit bug
            // See: https://github.com/rust-windowing/winit/issues/987
            // See: https://github.com/rust-windowing/winit/issues/1619
            glutin::event::Event::RedrawEventsCleared if cfg!(windows) => redraw(),
            glutin::event::Event::RedrawRequested(_) if !cfg!(windows) => redraw(),

            glutin::event::Event::WindowEvent { event, .. } => {
                if egui.is_quit_event(&event) {
                    *control_flow = glium::glutin::event_loop::ControlFlow::Exit;
                }

                egui.on_event(&event);

                display.gl_window().window().request_redraw(); // TODO: ask egui if the events warrants a repaint instead
            }

            _ => (),
        }
    });
}
