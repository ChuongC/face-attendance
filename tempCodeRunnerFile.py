                      log_attendance_to_django(name, similarity=sim)
                        except Exception as e:
                            logging.warning(f"Django sync failed for {name}: {e}")